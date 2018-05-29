function [bg_spatial,bg_temporal]=rank_1_factorization(Y,maxIter)
%% RANK_1_FACTORIZATION Rank-1-matrix-factorization of the movie Y
%
%       Y~bg_spatial*bg_temporal
%
% Input:
% Y                      movie
% max_iter               maximum Number of Iterations
%
% Output:
% bg_temporal            temporal component of the rank-1-factorization
% bg_spatial             spatial component of the rank-1-factorization
%
% This algorithm performs a form a block-wise gradient descent on the objective function
% $$L\\left( s,t \\right) = \\sum\_{\\text{ij}}^{}\\left( Y\_{\\text{ij}} - s\_{i}t\_{j} \\right)^{2}$$
% Here, s corresponds to *bg\_spatial*, and t corresponds to *bg\_temporal*.
% This can be seen when we calculate the gradient along s and t:
% *D*<sub>*s*</sub>*L* = 2(||*s*||<sub>2</sub><sup>2</sup>*s* − *s* \* *Y*)
% *D*<sub>*t*</sub>*L* = 2(||*t*||<sub>2</sub><sup>2</sup>*t* − *Y* \* *t*)
% and set them to zero. Between update we normalize the previously updated component. 
% This simplifies the code and leads to better performance in the general case of NNMF.

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
