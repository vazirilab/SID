function Xguess = fast_deconv(forwardFUN, backwardFUN, df, maxIter, opts)

if nargin<=5
    opts.lambda=0;
    opts.p=1;
    opts.mode='basic';
else
    if ~isfield(opts,'lambda')
        opts.lambda=0;
        opts.p=1;
        opts.mode='basic';
    end
end

disp('Running fast_deconv() with options:');
disp(opts);

Xguess=df;
if strcmp(opts.mode,'TV')
    if opts.p==2
        if length(opts.p)==1
            opts.p=opts.p*[1,1,1];
        end
        filt=zeros(3,3,3);
        filt(2,2,1)=opts.lambda(3);
        filt(2,1,2)=opts.lambda(2);
        filt(1,2,2)=opts.lambda(1);
        filt(2,2,3)=opts.lambda(3);
        filt(2,3,2)=opts.lambda(2);
        filt(3,2,2)=opts.lambda(1);
        filt(2,2,2)=-sum(opts.lambda);
        Q = @(projection) backwardFUN(forwardFUN(projection)) - convn(projection,filt,'same');
        df=Q(Xguess)-df + opts.lambda_;
        disp('!')
    elseif opts.p==1
        center=zeros(3,3,3);
        center(2,2,2)=1;
        up=zeros(3,3,3);
        up(2,2,3)=1;
        down=zeros(3,3,3);
        down(2,2,1)=1;
        left=zeros(3,3,3);
        left(1,2,2)=1;
        right=zeros(3,3,3);
        right(3,2,2)=1;
        front=zeros(3,3,3);
        front(2,1,2)=1;
        hind=zeros(3,3,3);
        hind(2,3,2)=1;
    end
elseif strcmp(opts.mode,'basic')
    Q = @(projection) backwardFUN(forwardFUN(projection));
    df=Q(Xguess)-df+opts.lambda_;  
end


for iter=1:maxIter
    tic;
    passive=max(Xguess>0,df<0);
    df_=passive.*df;
    if strcmp(opts.mode,'basic')
        alpha=sum(sum(sum(df_.^2,1),2),3)/sum(sum(sum(forwardFUN(df_).^2,1),2),3);
    else
        alpha=sum(sum(sum(df_.^2,1),2),3)/sum(sum(sum(df_.*Q(df_),1),2),3);
    end
    alpha(isnan(alpha))=0;
    alpha(isinf(alpha))=0;
    for id=1:length(alpha)
        df_(:,:,:,id)=Xguess(:,:,:,id)-alpha(id)*df_(:,:,:,id);
    end
    df_(df_<0)=0;
    df=df+Q(df_-Xguess);
    Xguess=df_;
    ttime = toc;
    disp(['  Iteration ' num2str(iter) '/' num2str(maxIter) ' took ' num2str(ttime) ' secs']);
end

end


