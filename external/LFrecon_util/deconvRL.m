function [Xguess, Xguess_history] = deconvRL(forwardFUN, backwardFUN, Htf, maxIter, Xguess, opts)
% ISRA with various penalties depending on opts
% opts.lambda ... multiplicative constant in front of penalty term
%                 if mode is 'TV' opts.lambda is expected to be 1x3 each component for the corresponding spatial direction's penalty.      
% opts.p      ... decides which p-norm is used in penalty term (p=1 or p==2
%                 for mode 'TV')
% opts.mode   ... decides whether the norm is applied to Xguess (if 'basic') or to the total variation of Xguess (if 'TV')

if nargout > 1
    Xguess_history = zeros([maxIter size(Xguess)], 'single');
end

if nargin < 6
    opts.lambda=0;
    opts.p=1;
    opts.mode='basic';
end

if strcmp(opts.mode,'TV')
    if opts.p==2
        filt=zeros(3,3,3);
        filt(2,2,1)=opts.lambda(3);
        filt(2,1,2)=opts.lambda(2);
        filt(1,2,2)=opts.lambda(1);
        filt(2,2,3)=opts.lambda(3);
        filt(2,3,2)=opts.lambda(2);
        filt(3,2,2)=opts.lambda(1);
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
end

for i = 1 : maxIter
    tic;
    HXguess = forwardFUN(Xguess);    
    if strcmp(opts.mode,'basic')
        if opts.lambda==0
            HXguessBack = backwardFUN(HXguess);
            errorBack = Htf ./HXguessBack;
        else
            if opts.p==1
                HXguessBack = backwardFUN(HXguess)+opts.lambda;
            else
                HXguessBack = backwardFUN(HXguess)+opts.lambda*opts.p*(Filter.*(Xguess.^(opts.p-1)));
            end
            errorBack = Htf ./HXguessBack;            
        end
    elseif strcmp(opts.mode,'TV');
        
        if opts.lambda==0
            HXguessBack = backwardFUN(HXguess);
            errorBack = Htf ./HXguessBack;
        else
            if opts.p==1
                dx=sign(convn(Xguess,center-left,'same'));
                dy=sign(convn(Xguess,center-front,'same'));
                dz=sign(convn(Xguess,center-down,'same'));
                HXguessBack = backwardFUN(HXguess)+opts.lambda(1)*dx+opts.lambda(2)*dy+opts.lambda(3)*dz+sum(opts.lambda);
                zaehler=Htf + opts.lambda(1)*convn(dx,right,'same')+opts.lambda(2)*convn(dy,hind,'same')+...
                    opts.lambda(3)*convn(dz,up,'same')+sum(opts.lambda);
                errorBack = zaehler./HXguessBack;
            elseif opts.p==2
                HXguessBack = backwardFUN(HXguess)+2*(opts.lambda(1)+opts.lambda(2)+opts.lambda(3))*Xguess;
                zaehler=Htf + convn(Xguess,filt,'same');
                errorBack = zaehler./HXguessBack;
            else
                disp('p>2 not supported');
                return;
            end
        end
    end
    errorBack(isinf(errorBack))=0;    
    Xguess = Xguess .* errorBack;
    Xguess(isnan(Xguess)) = 0;
    ttime = toc;
    disp(['  Iteration ' num2str(i) '/' num2str(maxIter) ' took ' num2str(ttime) ' secs']);
    if nargout > 1
        Xguess_history(i, :, :, :) = gather(Xguess);
    end
end
end