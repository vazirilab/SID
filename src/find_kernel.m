function [kernel, neur_rad] = find_kernel(Volume, border, neur_rad, native_focal_plane, axial, gpu_id)
%%
opts=struct;
opts.border = border;
if nargin==6
    opts.gpu_ids = gpu_id;
end
opts.axial=axial;
opts.neur_rad = neur_rad;
opts.native_focal_plane = native_focal_plane;

%%
rr{1}=Volume;
segmm=filter_recon(rr,opts);
segmm=max(segmm{1}-mean(segmm{1}(segmm{1}>0)),0);

beads=bwconncomp(segmm);

p=[];

for k=1:beads.NumObjects
    p(k)=norm(segmm(beads.PixelIdxList{k}));
end

if ~isempty(p)
    [~,n]=sort(-p);
    % n=n([1 min(length(n),11):length(n)]);
    
    n = n(11:max(length(n),10));
    
    for k=reshape(n,1,[])
        segmm(beads.PixelIdxList{k})=0;
    end
    
    centers=round(segment_component(segmm,0));
    
    id=(centers(:,1)>round(neur_rad)).*(centers(:,2)>round(neur_rad)).*...
        (centers(:,3)>round(neur_rad/axial));
    centers=centers(id>0,:);
    id=(centers(:,1)<size(segmm,1)-round(neur_rad)).*(centers(:,2)<size(segmm,2)...
        -round(neur_rad)).*(centers(:,3)<size(segmm,3)-round(neur_rad/axial));
    centers=centers(id>0,:);
    
    kernel=zeros(2*round(neur_rad*[1 1 1/axial])+1);
    for k=1:size(centers,1)
        vol = Volume(centers(k,1)-round(neur_rad):centers(k,1)+round(neur_rad),...
            centers(k,2)-round(neur_rad):centers(k,2)+round(neur_rad),...
            centers(k,3)-round(neur_rad/axial):centers(k,3)+round(neur_rad/axial));
        vol = vol/sum(vol(:));
        kernel=kernel + vol;
    end
    
    kernel = kernel/sum(kernel(:));
    p = squeeze(sum(sum(kernel,2),3));
    p = p + squeeze(sum(sum(kernel,1),3))';
    neur_rad = 3*sqrt(sum(p'/2.*[1:length(p)].^2)-sum(p'/2.*[1:length(p)])^2);
else
    kernel = 0;
    neur_rad = [];
end
end



