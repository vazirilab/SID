function [xx, Q, df, h]=fast_nnls(A,Y,opts,Q,df,h,temp)


%%
if  ~isfield(opts,'total')
    opts.total=0;
end

if nargin<3
    opts.gpu=false;
    opts.tol=1e-12;
end

if ~isfield(opts,'gpu')
    opts.gpu=false;
end

if ~isfield(opts,'tol_p')
    opts.tol_p=1;
end

if ~isfield(opts,'display')
    opts.display=false;
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

if ~isfield(opts,'lambda')
    opts.lambda = 0;
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

if opts.gpu
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
    %     tic
    x_=x;
    if opts.display
        disp(s)
    end
    s=s+1;
    df_=df.*passive;
    if opts.total
        alpha=sum(sum(df_.^2,1),2)./sum(sum(df_.*(Q*df_),1),2);
        alpha(isnan(alpha))=0;
        x_=x-df_*alpha;
        x_(x_<0)=0;
        df=df+Q*(x_-x);
        x=x_;
        
        if nargin==7
            passive=max(x>0,df<0).*temp;
        else
            passive=max(x>0,df<0);
        end
        
        if isfield(opts,'max_iter')
            if s>opts.max_iter
                disp('max number of iterations is reached');
                break;
            end
        end
    else
        alpha=sum(df_.^2,1)./sum(df_.*(Q*df_),1);
        alpha(isnan(alpha))=0;
        if opts.gpu
            x = gather(x);
            x_ = gather(x_);
            df_ = gather(df_);
            alpha = gather(alpha);
        end
        for ii=1:size(x,2)
            x_(:,ii)=x(:,ii)-df_(:,ii)*alpha(ii);
        end
        
        if opts.gpu
            x = gpuArray(x);
            x_ = gpuArray(x_);
        end
        x_(x_<0)=0;
        if mod(s,2)==1
            
            if opts.tol_p==2
                test=[test' log(sqrt(sum((x_-x).^2,1)))']';
            elseif opts.tol_p==1
                test=[test' log(sum(abs(x_-x),1))']';
            end
            
            
            if size(test,1)>opts.sample
                test=test(2:end,:);
            end
            if s>2*opts.sample
                k=Xi*test;
                ids=logical(max((max(passive)==0),(h_*exp(test(end,:))./(1-exp(k(1,:)))<opts.tol*size(x,1)*max(x,[],1)).*(sum(abs(X*k-test),1)<opts.tol_*opts.sample).*(exp(k(1,:))<1)));
            end
            
        else
            ids=[];
        end
        df=df+Q*(x_-x);
        
        if isfield(opts,'max_iter')
            if s>opts.max_iter
                ids=1:size(x,2);
                disp('max number of iterations is reached');
            end
        end
        x=x_;
        
        
        if nargin==7
            passive=max(x>0,df<0).*temp;
        else
            passive=max(x>0,df<0);
        end
        
        if max(ids(:))
            if opts.display
                disp(find(ids))
            end
            if opts.gpu
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
    %     toc
end

if opts.gpu
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



