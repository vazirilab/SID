function [X, G] = LS_nnls(A,Y,opts,G,x)
% 
if nargin<3
    opts =struct;
end

if ~isfield(opts,'display')
    opts.display = false;
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

h = A'*Y;

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

iter = 0;
test=[];
dn=[[1:2:2*opts.sample]', ones(opts.sample,1)];
Xi=inv(dn'*dn)*dn';
tic
while ~isempty(x)
    iter = iter + 1;
    x_ = gather(x);
    df = -h + G*x + opts.lambda;
    passive=max(x>0,df<0);
    df_=df.*passive;
    alpha=sum(df_.^2,1)./sum(df_.*(G*df_),1);
    alpha(isnan(alpha))=0;
    x = x - df_.*alpha;
    x(x<0) = 0;
    
    if iter>opts.max_iter
        ids=true(1,size(x,2));
        disp('max number of iterations is reached');
    else
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
toc

end