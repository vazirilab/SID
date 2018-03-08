function [bg_spatial,bg_temporal]=rank_1_factorization(Y,maxIter)

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