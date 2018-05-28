function [X, G] = LS_nnls(A,Y,opts,G,x)
% LS_NNLS is a solver for large non-negative least squares problems.
%
%                   argmin_{x>=0}||y-A(x)||_2^2
%
% Input:
% A...              Matrix corresponding to A in the formula above.
% x...              Matrix of solution vectros of the above problem. LS_nnls
%                   solves multiple nnls problems in parallel.
% Y...              Matrix of inhomogenities, each row represents one nnls
%                   problm.
% struct opts
% opts.display...   boolean, if true messages will be printed in console.
% opts.lambda...    lagrangian multiplier for L1 regularization
% opts.gpu_id...    ID of GPU to be used if GPU support is available.
% opts.use_std...   calculate least standard deviation instead of L2-norm
% opts.sample...    Read about convergence check below!
% opts.tol...       -
% opts.tol_...      -
%
% Output
% X...              Matrix of approximations of the solutions to the
%                   nnls problems. Each row is one solution.
% G...              Gram-matrix.
%
% Convergence check:
% LS_nnls checks for each of the nnls subproblem if a certain accuracy is
% reached. It does that no like usually by checking the norm of the
% gradient, but by guessing the convergence curve from the difference
% between two consecutive iterative solutions (Consec_error).
% It performs a linear fit of the log of Consec_error for opts.sample
% previous iterates and if the error between the linear fit and the log of
% the consecutive error is smaller than opts.tol_ the algorithm can use
% this information to accurately estimate the true error. If the true error
% is below opts.tol the algorithm stops for the nnls-sub-problem in
% question and puts the current solution into the output array.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<3
    opts =struct;
end

if ~isfield(opts,'display')
    opts.display = false;
end

if ~isfield(opts,'use_std')
    opts.use_std = false;
end

if ~isfield(opts,'lambda')
    opts.lambda = 0;
end

if ~isfield(opts,'gpu_id')
    opts.gpu_id = [];
end

if ~isfield(opts,'sample')
    opts.sample = 300;
end

if ~isfield(opts,'tol')
    opts.tol = 1e-7;
end

if ~isfield(opts,'tol_')
    opts.tol_ = 1e-2;
end

if ~isfield(opts,'max_iter')
    opts.max_iter = 2000;
end

h = A'*Y - opts.lambda;

if nargin<5
    x = zeros(size(A,2),size(Y,2));
    if nargin<4
        G = A'*A;
    elseif size(G,1)~=size(A,2)
        disp('G has wrong dimensions.')
        return
    end
elseif (size(x,1)~=size(A,2))||(size(x,2)~=size(Y,2))
    disp('x has wrong dimensions.')
    return
end
X = x;
if ~isempty(opts.gpu_id)
    gpuDevice(opts.gpu_id);
    x = gpuArray(x);
    h = gpuArray(h);
    G= gpuArray(G);
end
rds=1:size(Y,2);

if opts.use_std
    G = @(x) G*x - (G*sum(x,2))/size(Y,2);
    h = h - A'*sum(Y,2)/size(Y,2);
else
    G = @(x) G*x;
end

iter = 0;
test=[];
dn=[[1:2:2*opts.sample]', ones(opts.sample,1)];
Xi=inv(dn'*dn)*dn';
tic
while ~isempty(x)
    iter = iter + 1;
    x_ = gather(x);
    df = -h + G(x);
    passive=max(x>0,df<0);
    df_=df.*passive;
    alpha=sum(df_.^2,1)./sum(df_.*(G(df_)),1);
    alpha(isnan(alpha))=0;
    x = x - df_.*alpha;
    x(x<0) = 0;
    
    if iter>opts.max_iter
        ids=true(1,size(x,2));
        if opts.display
            disp('max number of iterations is reached');
        end
    elseif ~opts.use_std
        if mod(iter,2)==1
            test = [test' log((sum(gather(x)-x_,1).^2)./sum(x_.^2,1))'/2]';
            if size(test,1)>opts.sample
                test=test(2:end,:);
            end
            if iter>2*opts.sample
                k = Xi*test;
                ids = logical(max(((exp(test(end,:))./(1 - exp(k(1))))<opts.tol).*...
                    (sum(abs(dn*k-test),1)<opts.tol_*opts.sample).*...
                    (exp(k(1,:))<1),test(end,:)==-inf));
                %                 ids = logical(max((max(passive)==0),(exp(test(end,:))./(1-exp(k(1,:)))<opts.tol*size(x,1)*max(x,[],1)).*(sum(abs(dn*k-test),1)<opts.tol_*opts.sample).*(exp(k(1,:))<1)));
            else
                ids = false(1,size(x,2));
            end
        else
            ids=[];
        end
    else
        ids = [];
    end
    
    if max(ids(:))
        if ~isempty(opts.gpu_id)
            X(:,rds(ids)) = gather(x(:,ids));
        else
            X(:,rds(ids)) = x(:,ids);
        end
        rds = rds(~ids);
        x=x(:,~ids);
        h=h(:,~ids);
        test = test(:,~ids);
    end
    if opts.display
        disp(iter);
    end
end
if opts.display
    toc
end

end