function [centroid, ID, centers] = iterate_cluster(centers_cell, N, radius, dim)
% ITERATE_CLUSTER clusters the elements of the cell array centers_cell, 
% such that the clusters have a maximal radius according to input argument "radius"
% and each of their elements is from a different cell of centers_cell.
%
% Input:
% centers_cell         cell array of array of size [m,3]
% N                    number of iterations to perform
% radius               max radius of cluster
% dim                  vector of relative voxel edge lengths along the three dimensions

centroid =[];
for jj=1:size(centers_cell,2)
    centroid=[centroid' centers_cell{jj}']';
end

for nn=1:N
    centers=[];
    centroid_=[];
    centers_per_component_=centers_cell;
    for ii=1:size(centroid,1)
        centers{ii}=[]; %#ok<AGROW>
        for jj=1:size(centers_per_component_,2)
            if ~isempty(centers_per_component_{jj})
                rnd_ind=randperm(size(centers_cell{jj},1));
                [a, n] = min(sum((dim .* (centers_per_component_{jj}(rnd_ind,:) - centroid(ii,:))) .^2, 2));
                if a<radius^2
                    centers{ii}=[centers{ii}' centers_per_component_{jj}(rnd_ind(n),:)']';
                    centers_per_component_{jj}(rnd_ind(n),:)=inf*[1 1 1];
                end
            end
        end
        centroid_=[centroid_' mean(centers{ii},1)']';
    end
    idx=randperm(size(centroid_,1));
    centroid=centroid_(idx,:);
    %disp(size(centroid));
    %disp(nn);
end

[~,n] = sort(sum(centroid,2));

if nargout>1
    ID=false(size(centroid,1),size(centers_cell,2));
    for ii=1:size(centroid,1)
        for jj=1:size(centers_per_component_,2)
            if ~isempty(centers_per_component_{jj})
                a = min(sum((dim .* (centers_cell{jj} - centroid(ii,:))) .^2, 2));
                if a<radius^2
                    ID(ii,jj)=true;
                end
            end
        end
    end
    ID=ID(n,:);
end

centroid = centroid(n,:);
end
