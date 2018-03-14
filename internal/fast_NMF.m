function [S,T]=fast_NMF(Y,n,opts,T,S)
% FAST_NMF: the algorithm performs a non-negative matrix factorization
% (nnmf) on the movie Y, using an alternating convex search
% approach. Each of the updates is performed by gradient descent with
% exact line search.
%
% Input:
% n...                  rank of the nnmf.
% T...                  Initial condition for the temporal component of the nnmf.
%                       If T is not set, the algorithm will compute a first guess
%                       based on opts.ini.
% S...                  Initial guess for the spatial compontent of the nnmf.
%                       If S is not set, it is computed by S=Y*T
% struct opts:
% opts.lamb_spat...      lagrangian multiplier for L1 regularizer on S
% opts.lambda.temp...      lagrangian multiplier for L1 regularizer on T
% opts.lamb_corr...  lagrangian multiplier for L2 regularizer on
%                       corrcoef(T)-eye(size(S,2))
% opts.lamb_orth...   lagrangian multiplier for L2 regularizer on
%                       S'*S-eye(size(S,2))
% opts.lamb_TotVarSpat...     lagrangian multiplier for L2 regularizer on the
%                       total variation of the components of S.
%                       It is necessary that the size (height & width) of
%                       the linearized components of S is included, since
%                       the algorithm needs the reshape S accordingly.
% opts.ini...           Initialization method for T. opts.ini='pca' uses
%                       the first "n" principal components. opts.ini="rand"
%                       generates "n" smoothed random traces as
%                       initialization for T.
% opts.max_iter...      maximum number of iterations
% opts.lb...            Integer; Lower bound for the number of non-negative
%                       pixels per spatial component.
%%
if nargin<3
    opts.lamb_spat=0;
    opts.lamb_temp=0;
    opts.lamb_corr=0;
    opts.lamb_orth=0;
    opts.lamb_TotVarSpat=0;
    opts.ini='pca';
else
    if ~isfield(opts,'lamb_spat')
        opts.lamb_spat=0;
    end
    if ~isfield(opts,'lamb_temp')
        opts.lamb_temp=0;
    end
    if ~isfield(opts,'lamb_corr')
        opts.lamb_corr=0;
    end
    if ~isfield(opts,'lamb_orth')
        opts.lamb_orth=0;
    end
    if ~isfield(opts,'active')
        opts.active = ones(1,size(Y,1),'logical');
    end
    if ~isfield(opts,'lamb_TotVarSpat')
        opts.lamb_TotVarSpat=0;
    end
    if ~isfield(opts,'ini')
        opts.ini='pca';
    end
    if ~isfield(opts,'lb')
        opts.lb = 133;
    end
end

if opts.lamb_TotVarSpat
    if ~isfield(opts,'size')
        disp('The option lamb_TotVarSpat needs additional information: size of the image (opts.size)');
        return
    end
    laplace = zeros(1,3,3);
    laplace(1,2,2)=4;
    laplace(1,1,2)=-1;
    laplace(1,3,2)=-1;
    laplace(1,2,1)=-1;
    laplace(1,2,3)=-1;
end


opts.lamb_orth = opts.lamb_orth * norm(Y(:))*size(Y,2);
opts.lamb_spat = opts.lamb_spat * norm(Y(:))*size(Y,2);
opts.lamb_temp = opts.lamb_temp * norm(Y(:));
opts.lamb_corr = opts.lamb_corr * norm(Y(:));
opts.lamb_TotVarSpat = opts.lamb_TotVarSpat * norm(Y(:));

flag_t = false;
flag_s = false;

if ~n
    opts.max_iter=0;
end

if nargin<4
    if strcmp(opts.ini,'rand')
        T_0=rand(n,size(Y,2));
        T_0=conv2(T_0,ones(1,32),'same');
    else
        [T_0,~,~] = pca(Y(opts.active,:));
        T_0 = abs(T_0(:,1:n))';
    end
    T = T_0;
end

if nargin<5
    S=Y*T';
end

iter = 0;
while iter<opts.max_iter
    iter = iter + 1;
    T(isnan(T))=0;
    line = ~logical(sum(T,2));
    
    if max(line)
        T = T_0;
        S=Y*T';
        opts.lamb_spat = opts.lamb_spat/10;
        opts.lamb_temp = opts.lamb_temp/10;
        opts.lamb_corr = opts.lamb_corr/10;
        opts.lamb_orth = opts.lamb_orth/10;
        opts.lamb_TotVarSpat = opts.lamb_TotVarSpat/10;
        disp(opts);
        iter = 0;
        flag_t = true;
        disp('ZERO detected - ReInitializing');
        continue
    elseif flag_t
        T = T_0;
        S=Y*T';
        opts.lamb_spat = opts.lamb_spat/10;
        opts.lamb_temp = opts.lamb_temp/10;
        opts.lamb_corr = opts.lamb_corr/10;
        opts.lamb_orth = opts.lamb_orth/10;
        opts.lamb_TotVarSpat = opts.lamb_TotVarSpat/10;
        disp(opts);
        iter = 0;
        flag_t = false;
    end
    
    for u=1:size(T,1)
        platz = norm(T(u,:));
        T(u,:) = T(u,:)/platz;
        S(:,u) = S(:,u)*platz;
    end
    
    zsc = zscore(T);
    N =size(T,2);
    d = std(T,[],2);
    
    Q_S = S(opts.active,:)'*S(opts.active,:);
    q_T = S(opts.active,:)'*Y(opts.active,:);
    
    if opts.lamb_corr
        hilf = ((zsc*zsc')*zsc - zsc);
        hilf = diag(1./d)*(hilf - (diag(sum(hilf,2))*ones(size(T))/N + (diag(diag(hilf*zsc'))*zsc)/N));
        df_T = -q_T + Q_S*T + opts.lamb_temp + opts.lamb_corr*hilf;
    else
        df_T = -q_T + Q_S*T + opts.lamb_temp;
    end
    
    passive_T = max(T>0,df_T<0);
    
    df_T_ = passive_T.*df_T;
    
    alpha_T = sum(sum(df_T_.^2,1),2)/sum(sum(df_T_.*(Q_S*df_T_),1),2);
    
    alpha_T(isnan(alpha_T))=0;
    alpha_T(isinf(alpha_T))=0;
    if ~max(isnan(alpha_T(:)))
        T = T - alpha_T*df_T_;
    end
    
    T(T<0)=0;
    if opts.lamb_orth
        for u=1:size(T,1)
            platz = norm(S(:,u));
            T(u,:) = T(u,:)*platz;
            S(:,u) = S(:,u)/platz;
        end
    end
    
    S(isnan(S))=0;
    
    
    line = ~logical(sum(S,1).*(sum(S>0,1)>opts.lb));

    if max(line)
        T = T_0;
        S=Y*T';
        opts.lamb_spat = opts.lamb_spat/10;
        opts.lamb_temp = opts.lamb_temp/10;
        opts.lamb_corr = opts.lamb_corr/10;
        opts.lamb_orth = opts.lamb_orth/10;
        opts.lamb_TotVarSpat = opts.lamb_TotVarSpat/10;
        disp(opts);
        iter = 0;
        flag_s = true;
        disp('ZERO detected - ReInitializing');
        continue
    elseif flag_s
        T = T_0;
        S=Y*T';
        opts.lamb_spat = opts.lamb_spat/10;
        opts.lamb_temp = opts.lamb_temp/10;
        opts.lamb_corr = opts.lamb_corr/10;
        opts.lamb_orth = opts.lamb_orth/10;
        opts.lamb_TotVarSpat = opts.lamb_TotVarSpat/10;
        disp(opts);
        iter = 0;
        flag_s = false;
    end
    
    Q_T = T*T';
    q_S = Y*T';
    
    if opts.lamb_orth
        df_S = -q_S + S*Q_T + opts.lamb_spat + opts.lamb_orth*(S*(S'*S)-S);
    else
        df_S = -q_S + S*Q_T + opts.lamb_spat;
    end
    
    if opts.lamb_TotVarSpat
        df_S = df_S + opts.lamb_TotVarSpat*reshape(convn(reshape(S',n,opts.size(1),opts.size(2)),laplace,'same'),n,[])';
    end
    
    passive_S = max(S>0,df_S<0);
    
    df_S_ = passive_S.*df_S;
    
    alpha_S = sum(sum(df_S_.^2,1),2)/sum(sum(df_S_.*(df_S_*Q_T),1),2);
    alpha_S(isnan(alpha_S))=0;
    alpha_S(isinf(alpha_S))=0;
    if ~max(isnan(alpha_S(:)))
        S = S - alpha_S*df_S_;
    end
    
    if (max(df_S_(:))==0)&&(max(df_T_(:))==0)
        disp('Terminated!')
        return;
    end
    
    S(S<0)=0;
    
    disp(['Iteration ' num2str(iter) ' completed']);
end

for u=1:size(T,1)
    platz = norm(T(u,:));
    T(u,:) = T(u,:)/platz;
    S(:,u) = S(:,u)*platz;
end

end