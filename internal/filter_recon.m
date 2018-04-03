function segmm=filter_recon(recon, opts)

[~,u] = max([size(recon,1),size(recon,2)]);
poolobj = gcp('nocreate');

if nargin<2
    opts.border=[1 1 15];
    opts.NumWorkers=length(opts.gpu_ids);
end

if ~isfield(opts,'NumWorkers')
    if isfield(opts,'gpu_ids')&&(~isempty(opts.gpu_ids))
        opts.NumWorkers=length(opts.gpu_ids);
    else
        opts.NumWorkers=1;
    end
end

if ~isfield(opts,'border')
    opts.border=[1 1 15];
end

if isempty(poolobj)||(poolobj.NumWorkers~=opts.NumWorkers)
    delete(poolobj);
    poolobj=parpool(opts.NumWorkers);
end

if isfield(opts,'gpu_ids')&&(~isempty(opts.gpu_ids))
    n=length(opts.gpu_ids);
    gimp=opts.gpu_ids;
else
    n=1;
    gimp=-1;    
end
segmm=recon;
decay = (1/100)^(1/min(opts.border(3),2*opts.neur_rad/opts.axial));

for kk=1:opts.NumWorkers:size(recon,u)
    img=cell(opts.NumWorkers,1);
    si_V=cell(opts.NumWorkers,1);
    siz_I=cell(opts.NumWorkers,1);
    segm_=cell(opts.NumWorkers,1);
    for worker=1:min(opts.NumWorkers,size(recon,u)-(kk-1))
        k=kk+worker-1;        
        if opts.neur_rad<5
            [X,Y,Z]=meshgrid(1:2:2*size(recon{k},2)-1,1:2:2*size(recon{k},1)-1,[1:opts.native_focal_plane-1 opts.native_focal_plane+1:size(recon{k},3)]);
            [Xq,Yq,Zq]=meshgrid(1:1:2*size(recon{k},2)-1,1:1:2*size(recon{k},1)-1,1:size(recon{k},3));
            cellSize = 4*opts.neur_rad;
        else
            cellSize = 2*opts.neur_rad;
        end       
        if opts.neur_rad<5
            V=interp3(X,Y,Z,recon{k}(:,:,[1:opts.native_focal_plane-1 opts.native_focal_plane+1:size(recon{k},3)]),Xq,Yq,Zq);
        else
            V=recon{k};
        end
        si_V{worker}=size(V,3);
        I=zeros(size(V)+[0 0 2*opts.border(3)],'single');
        I(:,:,1+opts.border(3):si_V{worker}+opts.border(3))=single(V);
        for k=0:opts.border(3)-1
            I(:,:,opts.border(3)-k)=I(:,:,opts.border(3)+1-k)*decay;
            I(:,:,opts.border(3)+si_V{worker}+k)=I(:,:,opts.border(3)+si_V{worker}-1+k)*decay;
        end
        img{worker}=full(I/max(I(:)));
        siz_I{worker}=size(I);
    end
    parfor worker=1:min(opts.NumWorkers,size(recon,u)-(kk-1))
        filtered_Image_=band_pass_filter(img{worker}, cellSize, 3, gimp(mod(worker-1,n)+1),1.2);
        segm_{worker}=filtered_Image_(opts.border(1):size(filtered_Image_,1)-opts.border(1),opts.border(2):size(filtered_Image_,2)-opts.border(2),opts.border(3)+1:opts.border(3)+si_V{worker});
        if ~isempty(opts.gpu_ids)
            gpuDevice([]);
        end
    end
    for kp=1:min(opts.NumWorkers,size(recon,u)-(kk-1))
        filtered_Image=zeros(siz_I{kp}-[0 0 2*opts.border(3)]);
        filtered_Image(opts.border(1):siz_I{kp}(1)-opts.border(1),opts.border(2):siz_I{kp}(2)-opts.border(2),:)=segm_{kp};
        if opts.neur_rad<5
            segmm{kk+kp-1}=filtered_Image(1:2:end,1:2:end,:);
        else
            segmm{kk+kp-1}=filtered_Image;
        end        
    end
    disp(['Filtering of Volumes from ' num2str(kk) ' to ' ...
        num2str(min(kk+poolobj.NumWorkers-1,size(recon,u))) ' completed']);
end

for ix=1:size(recon,u)
    Vol = recon{ix}*0;
    Vol(opts.border(3):end-opts.border(3),opts.border(3):end-opts.border(3),:) = segmm{ix}(opts.border(3):end-opts.border(3),opts.border(3):end-opts.border(3),:);
    segmm{ix} = Vol/max(Vol(:));
end

end