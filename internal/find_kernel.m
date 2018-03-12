function kernel=find_kernel(Volume,Input)

rr{1}=Volume;
opts=struct;
opts.border = [1,1,15];
opts.gpu_ids = Input.gpu_ids;
opts.neur_rad = Input.neur_rad;
opts.native_focal_plane = Input.native_focal_plane;
segmm=filter_recon(rr,opts);
segmm=max(segmm{1}-mean(segmm{1}(segmm{1}>0))/2,0);

beads=bwconncomp(segmm);

for k=1:beads.NumObjects
    p(k)=norm(segmm(beads.PixelIdxList{k}));
end

[~,n]=sort(-p);
n=n([1 min(length(n),11):length(n)]);
for k=n
    segmm(beads.PixelIdxList{k})=0;
end

centers=round(segment_component(segmm,0));

id=(centers(:,1)>Input.neur_rad).*(centers(:,2)>Input.neur_rad).*...
    (centers(:,3)>Input.neur_rad/Input.axial);
centers=centers(id>0,:);
id=(centers(:,1)<size(segmm,1)-Input.neur_rad).*(centers(:,2)<size(segmm,1)-Input.neur_rad).*...
    (centers(:,3)<size(segmm,1)-Input.neur_rad/Input.axial);
centers=centers(id>0,:);


kernel=zeros(2*round(Input.neur_rad*[1 1 1/Input.axial])+1);
for k=1:size(centers,1)
    kernel=kernel+Volume(centers(k,1)-round(Input.neur_rad):centers(k,1)+round(Input.neur_rad),...
        centers(k,2)-round(Input.neur_rad):centers(k,2)+round(Input.neur_rad),...
        centers(k,3)-round(Input.neur_rad/Input.axial):centers(k,3)+round(Input.neur_rad/Input.axial));
end
end



