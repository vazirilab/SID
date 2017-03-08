function [S,T]=fast_NMF(Y,n,opts,T)

if n==0
    S=[];
    T=[];
else
    if nargin<4
        T=ones(n,size(Y,2));
        for k=1:n-1
            T(k+1,:)=sin(k*[0:size(Y,2)-1]/(size(Y,2)-1)*pi)+1;
        end
        if isfield(opts,'bg_temporal')
            T(end,:)=opts.bg_temporal;
        end
        
    end
    opts.warm_start=[];
    option=opts;
    option.max_iter=3;
    for iter=1:opts.max_iter
        option.lambda=opts.lambda;
        if iter>1
            option.warm_start=S;
        end
        S=fast_nnls(T',Y',option);
        S(isnan(S))=rand(size(S(isnan(S))));
        option.lambda=0;
        if iter>1
            option.warm_start=T;
        end
        T=fast_nnls(S',Y,option);
        T(isnan(T))=rand(size(T(isnan(T))));
        if mod(iter, 20) == 1
            fprintf([num2str(iter) ' ']);
        end
    end
end
fprintf('\n')
end