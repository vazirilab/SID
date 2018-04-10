function  [T,S] = initialize_nnmf(Y,n,opts)
% INITIALIZE_NNMF: algorithm initializes S and T for an nnmf according to
% the parameters in opts.
%
% Input:
% Y...              movie
% n...              Dimensions of nnmf
% struct opts:
% opts.ini_method   Initialization method:
%                   'rand' initialize T as smoothed random signal
%                          and initialize S as S = nnls(Y,T); 
%                   'pca' initialize T and S as the first 'n' principal
%                   components
% 
% Output:   
% T...              initial temporal components for nnmf
% S...              initial spatial components for nnmf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(opts.ini_method,'rand')
    T=conv2(rand(n,size(Y,2)),ones(1,32),'same');
    option=opts;
    if ~isfield(opts,'max_iter')
        option.max_iter=12;
    end
    option.lambda=opts.lamb_spat;
    S=LS_nnls(T',Y',option)';
elseif strcmp(opts.ini_method,'pca')
    [T,S,~] = pca(Y);
    T = abs(T(:,1:n))';
    S = abs(S(:,1:n));
end
end