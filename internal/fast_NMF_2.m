function [S,T]=fast_NMF_2(Y,n,opts,T,S)
%nnmf using block coordinate descent with exact line search
%%
if nargin<4
    T=rand(n,size(Y,2));
    T=conv2(T,ones(1,32),'same');
    if isfield(opts,'bg_temporal')
        T(end,:)=opts.bg_temporal;
    end
    
end
if nargin<3
    opts.tol=1e-6;
else
    if ~isfield(opts,'tol')
        opts.tol=1e-6;
    end
end

opts.warm_start=[];
option=opts;
option.max_iter=1;
option.total=1;

if ~n
    opts.max_iter=0;
end


opts.warm_start=[];
option=opts;
option.max_iter=1;
option.lambda=opts.lambda_s;

if nargin<5
    S=fast_nnls(T',Y',option)';
    S=S*inv(diag(sqrt(sum(S.^2,1))));
end
N=0;

for iter=1:opts.max_iter + N

    line = ~logical(sum(T,2));
    
    if max(line)
        T(line,:) = rand(sum(line),size(T,2));
        N=N+400;
    end
    
    Q_S = S'*S;
    q_T = S'*Y;   
    df_T = -q_T + Q_S*T + opts.lambda_t;    
    passive_T = max(T>0,df_T<0);    
    df_T_ = passive_T.*df_T;   
    alpha_T = sum(sum(df_T_.^2,1),2)/sum(sum(df_T_.*(Q_S*df_T_),1),2);
        
    if ~max(isnan(alpha_T(:)))
        T = T - alpha_T*df_T_;
    end
    
    T(T<0)=0;
    line = ~logical(sum(S,1));
    if max(line)
        S(:,line) = rand(size(S,1),sum(line));
        N=N+400;
    end   
    Q_T = T*T';
    q_S = Y*T';  
    df_S = -q_S + S*Q_T + opts.lambda_s;      
    passive_S = max(S>0,df_S<0);   
    df_S_ = passive_S.*df_S;    
    alpha_S = sum(sum(df_S_.^2,1),2)/sum(sum(df_S_.*(df_S_*Q_T),1),2);      
    if ~max(isnan(alpha_S(:)))
        S = S - alpha_S*df_S_;
    end   
    if (max(df_S_(:))==0)&&(max(df_T_(:))==0)
        return;
    end    
    S(S<0)=0;   
    fprintf([num2str(iter) ' ']);
end
fprintf('\n')
end