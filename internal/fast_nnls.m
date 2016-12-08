function [xx, Q, df, H]=fast_nnls(A,Y,opts,Q,df,H)


%%
if nargin<3
    opts.gpu='off';
    opts.tol=1e-12;
end

if ~isfield(opts,'gpu')
    opts.gpu='off';
end

if ~isfield(opts,'display')
    opts.display='off';
end

if ~isfield(opts,'tol')
    opts.tol=1e-12;
end
%%
if nargin<=3
    H=full(A'*A);
    df=-diag(1./sqrt(diag(H)))*(A'*Y);
    Q=H./sqrt(diag(H)*diag(H)');
else
    if isempty(df)
        df=-diag(1./sqrt(diag(H)))*(A'*Y);
    end
end

x=zeros(size(A,2),size(Y,2));

if strcmp(opts.gpu,'on')
    gpu = gpuDevice(opts.gpu_ids);
    x=gpuArray(x);
    Q=gpuArray(Q);
    df=gpuArray(df);
end

passive=max(x>0,df<0);
rds=1:size(Y,2);
xx=ones(size(x));
s=1;
while ~isempty(x)
    if strcmp(opts.display,'on')
        disp(s)
        s=s+1;
    end
    df_=df.*passive;
    alpha=sum(df_.^2,1)./diag(((df_)'*Q*df_))';
    alpha(isnan(alpha))=0;
    x_=x-df_*diag(alpha);
    x_(x_<0)=0;
    df=df+Q*(x_-x);
    x=x_;
    passive=max(x>0,df<0);
    ids=max((max(passive)==0),(sum(df_.^2,1)<opts.tol));
    if max(ids(:))
        disp(find(ids))
        if strcmp(opts.gpu,'on')
            xx(:,rds(ids))=gather(x(:,ids));
        else
            xx(:,rds(ids))=x(:,ids);
        end
        rds=rds(~ids);
        x=x(:,~ids);
        df=df(:,~ids);
        passive=passive(:,~ids);
    end
end

if strcmp(opts.gpu,'on')
    if nargout>1
        Q=gather(Q);
    end
    if ~isempty(H)
        xx=gather(diag(1./(sqrt(diag(H))))*xx);
    else
        xx=gather(xx);
    end
    clear gpu;
    gpuDevice([]);
else
    if ~isempty(H)
        xx=diag(1./(sqrt(diag(H))))*xx;
    end
end

end



