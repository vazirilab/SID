function [vol,ker] = total_deconv(Volume,opts)

if nargin<2
    opts.max_iter=8;
    opts.size=[13,13,5];
    opts.thres=0.08;
else
    if ~isfield(opts,'max_iter')
        opts.max_iter=8;
    end
    if ~isfield(opts,'size')
        opts.size=[13,13,5];
    end
    if ~isfield(opts,'thres')
        opts.thres=0.08;
    end
end

if ~isempty(opts.gpu_ids)
    opts.gpu=1;
    gpuDevice(opts.gpu_ids(1));
end

vol = zeros(size(Volume) + 2*(opts.size-1)/2);
vol((opts.size(1)-1)/2+1:end-(opts.size(1)-1)/2,(opts.size(2)-1)/2+1:end-(opts.size(2)-1)/2,(opts.size(3)-1)/2+1:end-(opts.size(3)-1)/2) = Volume;
Volume=vol;
vol = vol/max(vol(:));

ker = fspecial3('gaussian',opts.size);

for iter=1:opts.max_iter
    
    centers=[];
    vol = vol/max(vol(:));
    vol=max(vol-opts.thres,0);
    B=reshape(vol,[],1);
    beads=bwconncomp(imregionalmax(vol));
    for k=1:beads.NumObjects
        qu=B(beads.PixelIdxList{1,k});
        q=sum(B(beads.PixelIdxList{1,k}));
        [a,b,c]=ind2sub(size(vol),beads.PixelIdxList{1,k});
        centers(k,:)=([a,b,c]'*qu/q)';
    end
    centers = round(centers);
    vol=vol*0;
    for ii=1:size(centers,1)
        vol((centers(ii,1)),(centers(ii,2)),(centers(ii,3)))=1;
    end
    
    for id=1:size(centers,1)
        ker = ker + Volume(centers(id,1)-(opts.size(1)-1)/2:centers(id,1)+(opts.size(1)-1)/2,...
            centers(id,2)-(opts.size(2)-1)/2:centers(id,2)+(opts.size(2)-1)/2,...
            centers(id,3)-(opts.size(3)-1)/2:centers(id,3)+(opts.size(3)-1)/2);
    end
    
    ker = ker/norm(ker(:));
    if opts.gpu
    Volume = gpuArray(Volume);
    
    ker=gpuArray(ker);
    end
    for iter_=1:22
        if iter_==1
            vol_ = convn(Volume,ker(size(ker,1):-1:1,size(ker,2):-1:1,size(ker,3):-1:1),'same');
            vol = vol_;
        end
        vol = vol .* (vol_)./(convn(convn(vol,ker,'same'),ker(size(ker,1):-1:1,size(ker,2):-1:1,size(ker,3):-1:1),'same'));
        vol(isnan(vol))=0;
        disp(iter_);
    end
    if opts.gpu
    Volume = gather(Volume);
    vol = gather(vol);
    ker = gather(ker);
    end
    disp(['Iteration ' num2str(iter) ' completed']);
end



