function [bg_spatial,bg_temporal]=rank_1_factorization(Y,maxIter)
% RANK_1_FACTORIZATION Rank-1-matrix-factorization of the movie Y
%
%       Y~bg_spatial*bg_temporal
%
% Input:
% Y...                      movie
% max_iter...               maximum Number of Iterations
%
% Output:
% bg_temporal...            temporal component of the rank-1-factorization
% bg_spatial...             spatial component of the rank-1-factorization

if nargin<2
    maxIter=1;
end

bg_spatial = ones(size(Y,1),1)/sqrt(size(Y,1));
for iter=1:maxIter
    bg_temporal = bg_spatial'*Y;
    bg_temporal = bg_temporal/norm(bg_temporal(:));
    bg_spatial = Y*bg_temporal';
    if iter<maxIter
        bg_spatial = bg_spatial/norm(bg_spatial(:));
    end
end


end