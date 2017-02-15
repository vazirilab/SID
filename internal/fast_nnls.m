function [xx, Q, df, h]=fast_nnls(A,Y,opts,Q,df,h,temp)


%%
if nargin<3
    opts.gpu='off';
    opts.tol=1e-12;
end

if ~isfield(opts,'gpu')
    opts.gpu='off';
end

if ~isfield(opts,'tol_p')
    opts.tol_p=1;
end

if ~isfield(opts,'display')
    opts.display='off';
end

if ~isfield(opts,'tol')
    opts.tol=1e-6;
end

if ~isfield(opts,'tol_')
    opts.tol_=1e-1;
end

if ~isfield(opts,'sample')
    opts.sample=30;
end

if ~isfield(opts,'warm_start')
    opts.warm_start=[];
end
%%
if nargin<=3
    Q=A'*A;
    h=1./sqrt(diag(Q));
    Q=Q./sqrt(diag(Q)*diag(Q)');
    df=diag(h)*(-A'*Y + opts.lambda);
else
    if isempty(df)
        df=diag(h)*(-A'*Y + opts.lambda);
    end
end

if isempty(opts.warm_start)
    x=zeros(size(A,2),size(Y,2));
elseif opts.warm_start==1
    x=inv(Q)*(-df);
    x(x<0)=0;
    df=df+Q*x;
    disp('warm_start');
else
    x=diag(1./h)*opts.warm_start;
    df=df+Q*x;
end

if strcmp(opts.gpu,'on')
    gpu = gpuDevice(opts.gpu_ids);
    x=gpuArray(x);
    Q=gpuArray(Q);
    df=gpuArray(df);
end

if nargin==7
    passive=max(x>0,df<0).*temp;
else
    passive=max(x>0,df<0);
end


h_=max(h);
rds=1:size(Y,2);
xx=ones(size(x));
s=1;
test=[];
X=[[1:2:2*opts.sample]', ones(opts.sample,1)];
Xi=inv(X'*X)*X';

while ~isempty(x)
    x_=x;
    if strcmp(opts.display,'on')
        disp(s)
    end
    s=s+1;
    df_=df.*passive;
    alpha=sum(df_.^2,1)./sum(df_.*(Q*df_),1);%diag(((df_)'*Q*df_))';
    alpha(isnan(alpha))=0;
    for ii=1:size(x,2)
        x_(:,ii)=x(:,ii)-df_(:,ii)*alpha(ii);
    end   
%     x_=x-df_*diag(alpha);
    x_(x_<0)=0;
    df=df+Q*(x_-x);
    
    if mod(s,2)==1
        if opts.tol_p==2
            test=[test' log(sqrt(sum((x_-x).^2,1)))']';
        elseif opts.tol_p==1
            test=[test' log(sqrt(sum(abs(x_-x),1)))']';
        end

        if size(test,1)>opts.sample
            test=test(2:end,:);
        end
        if s>2*opts.sample
            k=Xi*test;
            ids=logical(max((max(passive)==0),(h_*exp(test(end,:))./(1-exp(k(1,:)))<opts.tol*size(x,1)*max(x,[],1)).*(sum(abs(X*k-test),1)<opts.tol_*opts.sample).*(exp(k(1,:))<1)));       
        end
        if isfield(opts,'max_iter')
            if s>opts.max_iter
                ids=1:size(x,2);
                disp('max number of iterations is reached');
            end
        end
    else
        ids=[];
    end
    x=x_;
    passive=max(x>0,df<0);
    
    
    if max(ids(:))
        if strcmp(opts.display,'on')
            disp(find(ids))
        end
        if strcmp(opts.gpu,'on')
            xx(:,rds(ids))=gather(x(:,ids));
        else
            xx(:,rds(ids))=x(:,ids);
        end
        rds=rds(~ids);
        x=x(:,~ids);
        df=df(:,~ids);
        passive=passive(:,~ids);
        test=test(:,~ids);
        if nargin==7
            temp=temp(:,~ids);
        end
    end
end

if strcmp(opts.gpu,'on')
    if nargout>1
        Q=gather(Q);
    end
    if ~isempty(h)
        xx=gather(diag(h)*xx);
    else
        xx=gather(xx);
    end
    clear gpu;
    gpuDevice([]);
else
    if ~isempty(h)
        xx=diag(h)*xx;
    end
end

end



