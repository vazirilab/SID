function [centroid, ID, centers]=iterate_cluster(centers_cell,N,radius,dim)
% ITERATE_CLUSTER: Algorithm clusters the elements of the cell array
% centers_cell, such that the clusters have a maximal size of
% radius and each of their elements is from a different cell of
% centers_cell.
%
% Input:
% centers_cell...   cell array of array of size [m,3]
centroid =[];
for jj=1:size(centers_cell,2)
    centroid=[centroid' centers_cell{jj}']';
end


for nn=1:N
    centers=[];
    centroid_=[];
    centers_per_component_=centers_cell;
    for ii=1:size(centroid,1)
        centers{ii}=[];
        for jj=1:size(centers_per_component_,2)
            rnd_ind=randperm(size(centers_cell{jj},1));
            [a,n]=min(sum((dim.*(centers_per_component_{jj}(rnd_ind,:)-[centroid(ii,:)])).^2,2));
            if a<radius^2
                centers{ii}=[centers{ii}' centers_per_component_{jj}(rnd_ind(n),:)']';
                centers_per_component_{jj}(rnd_ind(n),:)=inf*[1 1 1];
            end
        end
        centroid_=[centroid_' mean(centers{ii},1)']';
    end
    idx=randperm(size(centroid_,1));
    centroid=centroid_(idx,:);
    size(centroid)
    disp(nn);
end

[~,n] = sort(sum(centroid,2));

if nargout>1
    ID=false(size(centroid,1),size(centers_cell,2));
    for ii=1:size(centroid,1)
        for jj=1:size(centers_per_component_,2)
            a=min(sum((dim.*(centers_cell{jj}-[centroid(ii,:)])).^2,2));
            if a<radius^2
                ID(ii,jj)=true;
            end
        end
    end
    ID=ID(n,:);
end


centroid = centroid(n,:);

end