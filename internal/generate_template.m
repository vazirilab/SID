function template=generate_template(centers,H,std_image,thres,shape)
% GENERATE_TEMPLATE computes for each center of a putative neuron its 
% template, a binary image fixing the area, where the LFM signature of the 
%putative neuron is to be expected.
%
% Input: 
% centers...        centroid of the putative neurons
% H...              point spread function of the LFM
% std_image...      standard deviation image of the LFM_movie
% thres...          threshold for the estimation of the radius in which an
%                   LFM signature is to be expected. The smaller 'thres'
%                   the greater the radius.
% shape...          boolean; if true the shape of the area, where the LFM 
%                   signature of the putative neuron is to be expected, has
%                   the shape of a disk, otherwise the shape of a square.
%
% Output:
% template...       Library of templates of the putative neurons

if nargin<4
    thres=0.03;
    shape=true;
end

if nargin<5
    shape=true;
end

disp('Generate template');

radius=squeeze(mean(mean(H,3),4));
for k=1:size(radius,3)
    radius(:,:,k)=radius(:,:,k)/max(reshape(radius(:,:,k),1,[]));
end
radius=sum(squeeze(mean(radius>thres,2))>0,1)/2;

if shape
    [X,Y]=meshgrid(1:size(std_image,2),1:size(std_image,1));
end

template=false([size(centers,1),size(std_image)]);
for k=1:size(centers,1)
    tic
    if shape
        template(k,:) = reshape(((Y-centers(k,1)).^2 + (X-centers(k,2)).^2 )<...
            radius(round(centers(k,3)))^2,1,[]);
    else
        tmp=false(size(std_image));
        tmp(max(1,round(centers(k,1))-radius(round(centers(k,3)))):...
            min(size(std_image,1),round(centers(k,1))+radius(round...
            (centers(k,3)))),max(1,round(centers(k,2))-radius(round(...
            centers(k,3)))):min(size(std_image,1),round(centers(k,2))...
            +radius(round(centers(k,3)))))=true;
        template(k,:) = tmp(:);
    end
    toc
end
disp([num2str(size(centers,1)) ' templates generated']);

end