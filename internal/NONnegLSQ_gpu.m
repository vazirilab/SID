function [x, B]=NONnegLSQ_gpu(A_sparse,A_full,y,template,opts, B)
%
% solves argmin_(x>=0) ||[A_sparse; A_full]*x-y||_2 + opts.lambda*||x||_opts.p
%
% convergences is tested by assuming exponential convergence of the
% algorithm, testing of the exponential model and terminating once the
% following terms are expected to be less than opts.tol away from the
% current term.
%
% opts.max_iter...maximal number of iterations
% opts.tol........tolerance for
% opts.tol_.......tolerance derivative of final step size
% opts.display....display the progress of the algorithm
% opts.checkNUM...Number of iterations final step size needs to be
% contained
% opts.sample.....number of steps between halting checks
% opts.anti.......if ~= 0 the adjoint problem is solved instead (Y'=A'*x')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(opts,'gpu')
    opts.gpu='on';
end

if ~isfield(opts,'gpu_ids')
    opts.gpu_ids=1;
end

if ~isfield(opts,'anti')
    opts.anti=0;
end

if ~isfield(opts,'max_iter')
    opts.max_iter=100000;
end

if ~isfield(opts,'tol')
    opts.tol = 1e-3;
end

if ~isfield(opts,'tol_')
    opts.tol_=1e-3;
end

if ~isfield(opts,'display')
    opts.display='off';
end

if ~isfield(opts,'sample')
    opts.sample=1000;
end

if ~isfield(opts,'skip')
    opts.skip=0;
end

if ~isfield(opts,'wait')
    opts.wait=opts.sample;
end

if ~isfield(opts,'lambda')
    opts.lambda=0;
end

if ~isfield(opts,'p')
    opts.p=1;
end

if ~isfield(opts,'checkNUM')
    opts.checkNUM=100;
end

%%
if strcmp(opts.gpu,'on')
    gpu = gpuDevice(opts.gpu_ids);
end

if opts.anti==0
    if isfield(opts,'weigths')
        A_sparse=diag(opts.weights)*A_sparse;
        A_full=diag(opts.weights)*A_full;
    end
    
    if nargin<6
        B_sparse=full(A_sparse'*A_sparse);
        B_full=A_full'*A_full;
        
        if (~isempty(A_sparse))&&(~isempty(A_full))
            B_mix=full(A_sparse'*A_full);
        else
            B_mix=[];
        end
    end
    
    x_sparse=[];
    x_full=[];
    
    if ~isempty(A_sparse)
        x_sparse=full(A_sparse'*y);
    end
    
    if ~isempty(A_full)
        x_full=A_full'*y;
    end
else
    if isfield(opts,'weigths')
        A_sparse=A_sparse*diag(opts.weights);
        A_full=A_full*diag(opts.weights);
    end
    
    if nargin<6
        B_sparse=full(A_sparse*A_sparse');
        B_full=A_full*A_full';
        
        if (~isempty(A_sparse))&&(~isempty(A_full))
            B_mix=full(A_sparse*A_full');
        else
            B_mix=[];
        end
    end
    
    x_sparse=[];
    x_full=[];
    
    if ~isempty(A_sparse)
        x_sparse=full((y*A_sparse')');
    end
    
    if ~isempty(A_full)
        x_full=(y*A_full')';
    end
end

if nargin<6
    if strcmp(opts.gpu,'on')
        B=gpuArray([B_sparse B_mix; B_mix' B_full]);
    else
        B=sparse([B_sparse B_mix; B_mix' B_full]);
    end
else
    if strcmp(opts.gpu,'on')
        B=gpuArray(B);
    end
end

if isempty(template)
    if strcmp(opts.gpu,'on')
        x_=gpuArray([x_sparse; x_full]);
    else
        x_=sparse([x_sparse; x_full]);
    end
else
    if strcmp(opts.gpu,'on')
        x_=gpuArray(full(template.*[x_sparse; x_full]));
    else
        x_=sparse(full(template.*[x_sparse; x_full]));
    end
end

clear B_sparse;
clear B_full;
clear B_mix;
clear x_sparse;
clear x_full;

%%

if ~isfield(opts,'penalty')
    opts.penalty=1;
end

xx=x_;

data_size=size(x_,2);

if opts.lambda==0
    for iter=1:opts.max_iter
        if isempty(template)
            error=x_./(B*xx);
        else
            error=max(x_./(B*xx),0);
        end
        %%
        if iter==1
            err_=ones(data_size,opts.max_iter,1);
            active=[1:data_size];
            in=active;
            passive=[];
            time=[];
            x=ones(size(xx));
        end
        err_(:,iter)=gather(sqrt(sum((xx(:,active)-error(:,active).*xx(:,active)).^2,1)));
        cook=gather(sqrt(sum(xx(:,active).^2,1)));
        if (mod(iter,opts.wait)==0)&&(iter>opts.wait)&&((~isempty(err_))||(min(time-iter)<=0))
            [active,active_, passive,in,new,out,out_,err_,time]=quality_manager(time,err_,active,passive,in,iter-1,cook,opts);
            xx=xx.*error;
            if ~isempty(out)
                x(:,out)=gather(xx(:,out_));
                xx=xx(:,new);
                x_=x_(:,new);
            end
            err_=err_(active_,:);
        else
            xx=xx.*error;
        end
        if length(in)<=data_size*opts.tol*opts.skip            % Break when residual data is small enough.
            break;
        end
        %%
        if strcmp(opts.display,'on')
            fprintf([num2str(iter) ' ']);
        end
    end
else
    for iter=1:opts.max_iter
        
        if (opts.p==1)&&(opts.lambda~=0)
            constr=diag(opts.penalty)*(xx*0+1);
        elseif (opts.p==0)&&(opts.lambda~=0)
            constr=convn(full(xx),[-1; 2; -1],'same');
        else
            constr = opts.p*((diag(opts.penalty)*x_).^(opts.p-1));
        end
        
        
        if isempty(template)
            error=x_./(B*xx+opts.lambda*constr);
        else
            error=max(x_./(B*xx+opts.lambda*constr),0);
        end
        %%
        if iter==1
            err_=ones(data_size,opts.max_iter,1);
            active=[1:data_size];
            in=active;
            passive=[];
            time=[];
            x=ones(size(xx));
        end
        err_(:,iter)=gather(sqrt(sum((xx(:,active)-error(:,active).*xx(:,active)).^2,1)));
        cook=gather(sqrt(sum(xx(:,active).^2,1)));
        if (mod(iter,opts.wait)==0)&&(iter>opts.wait)&&((~isempty(err_))||(min(time-iter)<=0))
            [active,active_, passive,in,new,out,out_,err_,time]=quality_manager(time,err_,active,passive,in,iter-1,cook,opts);
            xx=xx.*error;
            if ~isempty(out)
                x(:,out)=gather(xx(:,out_));
                xx=xx(:,new);
                x_=x_(:,new);
            end
            err_=err_(active_,:);
        else
            xx=xx.*error;
        end
        
        if length(in)<=data_size*opts.tol*opts.skip         % Break when residual data is small enough. isempty(in)||
            break;
        end
        %%
        if strcmp(opts.display,'on')
            fprintf([num2str(iter) ' ']);
        end
    end
end
disp(in(active));
disp(in(passive));
disp(time);
% figure;imagesc(zscore(x')')
if ~isempty(in)
    x(:,in)=gather(xx);
    in=[];
end
% figure;imagesc(zscore(x')')
x(isnan(x))=0;

% if ~isempty(err_)
%   figure;plot(log(err_(ceil(rand(1)*size(err_,1)),:)));
% end

if strcmp(opts.gpu,'on')
    if nargout==1
        gpu.reset();
        clear gpu
        gpuDevice([]);
    else
        B=gather(B);
    end
end

end


%%% error=max(asfasdf,0) fuehrt zu fehlern falls A negative eintraege hat
%%% xx=max(xx,0) ist wohl notwendig