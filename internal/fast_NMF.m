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
%                       based on opts.ini_method.
% S...                  Initial guess for the spatial compontent of the nnmf.
%                       If S is not set, it is computed by S=Y*T
% struct opts:
% opts.lamb_spat...     lagrangian multiplier for L1 regularizer on S
% opts.lambda.temp...   lagrangian multiplier for L1 regularizer on T
% opts.lamb_corr...     lagrangian multiplier for L2 regularizer on
%                       corrcoef(T)-eye(size(S,2))
% opts.lamb_orth...     lagrangian multiplier for L2 regularizer on
%                       S'*S-eye(size(S,2))
% opts.lamb_spat_TV...  lagrangian multiplier for L2 regularizer on the
%                       total variation of the components of S.
%                       It is necessary that the size (height & width) of
%                       the linearized components of S is included, since
%                       the algorithm needs the reshape S accordingly.
% opts.lamb_temp_TV...  lagrangian multiplier for L2 regularizer on the 
%                       Total Variation of the components of T.
% opts.ini_method...    Initialization method for T. opts.ini='pca' uses
%                       the first "n" principal components. opts.ini="rand"
%                       generates "n" smoothed random traces as
%                       initialization for T.
% opts.max_iter...      maximum number of iterations
% substruct opts.xval   see help of function xval
% opts.diagnostic       Generates a figure during runtime that displays the
%                       first ten components of S and T, as well as the L2
%                       error and the S'S reg-error.
% opts.pointwise...     Boolean; if true the updates will be performed
%                       pointwise, that means in every pixel and frame
%                       independently, otherwise they will be performed on
%                       the entire S or T.
%
% Ouput:
% S...                  Spatial components of the nnmf
% T...                  Temporal components of the nnmf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<3
    opts.lamb_spat=0;
    opts.lamb_temp=0;
    opts.lamb_corr=0;
    opts.lamb_orth=0;
    opts.lamb_spat_TV=0;
    opts.ini_method='pca';
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
    if ~isfield(opts,'lamb_spat_TV')
        opts.lamb_spat_TV=0;
    end
    if ~isfield(opts,'lamb_temp_TV')
        opts.lamb_temp_TV=0;
    end
    if ~isfield(opts,'ini_method')
        opts.ini_method='pca';
    end
    if ~isfield(opts,'diagnostic')
        opts.diagnostic=false;
    end
    if ~isfield(opts,'pointwise')
        opts.pointwise=false;
    end
end

if opts.lamb_spat_TV
    if ~isfield(opts,'size')
        disp('The option lamb_spat_TV needs additional information: size of the image (opts.size)');
        return
    end
    opts.laplace = zeros(1,3,3);
    opts.laplace(1,2,2)=4;
    opts.laplace(1,1,2)=-1;
    opts.laplace(1,3,2)=-1;
    opts.laplace(1,2,1)=-1;
    opts.laplace(1,2,3)=-1;
end

nrm=norm(Y(:));
opts.lamb_orth = opts.lamb_orth*nrm;
opts.lamb_corr = opts.lamb_corr*nrm;
opts.lamb_spat = opts.lamb_spat*nrm;
opts.lamb_temp = opts.lamb_temp*nrm;
opts.lamb_spat_TV = opts.lamb_spat_TV*nrm;
opts.lamb_temp_TV = opts.lamb_temp_TV*nrm;

if ~n
    opts.max_iter=0;
end

if nargin<4
    [T_0,S_0] = initialize_nnmf(Y,n,opts);
end

if nargin<5
    option=opts;
    if ~isfield(opts,'max_iter')
        option.max_iter=12;
    end
    option.lambda=opts.lamb_spat;
    S_0=LS_nnls(T_0',Y',option)';
end
%%
if opts.lamb_orth
    for u=1:size(T_0,1)
        platz = norm(S_0(:,u));
        T_0(u,:) = T_0(u,:)*platz;
        S_0(:,u) = S_0(:,u)/platz;
    end
    opts.hilf = ones(n);
    opts.hilf(1:end,1)=0;
    opts.hilf(1,1:end)=0;
    opts.hilf(eye(n)>0)=0;
end

if isfield(opts,'xval')
    opts=xval(Y,n,opts);
end
T = T_0;
S =S_0;


P=[];
E=[];
for iter=1:opts.max_iter
    [S,T]=T_update(Y,T,S,opts);
    [S,T]=S_update(Y,S,T,opts);
    if opts.diagnostic
        E(iter)=norm(reshape(std(Y-S*T,1,2),1,[]));
        P(iter)=norm(reshape(S'*S-eye(size(S,2)),1,[]));
        figure(12);
        plot(E);
        figure(13);
        plot(P);
        figure(14);
        imagesc(S'*S);
    end
    disp(['Iteration ' num2str(iter) ' completed']);
end

for u=1:size(T,1)
    platz = norm(T(u,:));
    T(u,:) = T(u,:)/platz;
    S(:,u) = S(:,u)*platz;
end

end