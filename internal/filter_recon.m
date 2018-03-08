function segmm=filter_recon(recon,Input,opts)


poolobj = gcp('nocreate');

if nargin<3
    opts.border=[1 1 15];
    opts.NumWorkers=length(Input.gpu_ids);
end

if ~isfield(opts,'NumWorkers')
    if isfield(Input,'gpu_ids')
        opts.NumWorkers=length(Input.gpu_ids);
    else
        opts.NumWorkers=1;
    end
end

if ~isfield(opts,'border')
    opts.border=[1 1 15];
end

if isempty(poolobj)||(poolobj.NumWorkers~=opts.NumWorkers)
    delete(poolobj);
    parpool(opts.NumWorkers);
end

if isfield(Input,'gpu_ids')
    n=length(Input.gpu_ids);
    gimp=Input.gpu_ids;
else
    n=1;
    gimp=-1;    
end
segmm=recon;
gpu = ~isempty(Input.gpu_ids);


for kk=1:opts.NumWorkers:size(recon,2)
    img=cell(opts.NumWorkers,1);
    si_V=cell(opts.NumWorkers,1);
    siz_I=cell(opts.NumWorkers,1);
    segm_=cell(opts.NumWorkers,1);
    for worker=1:min(opts.NumWorkers,size(recon,2)-(kk-1))
        k=kk+worker-1;        
        if Input.neur_rad<7
            [X,Y,Z]=meshgrid(1:2:2*size(recon{k},2)-1,1:2:2*size(recon{k},1)-1,[1:Input.native_focal_plane-1 Input.native_focal_plane+1:size(recon{k},3)]);
            [Xq,Yq,Zq]=meshgrid(1:1:2*size(recon{k},2)-1,1:1:2*size(recon{k},1)-1,1:size(recon{k},3));
            cellSize = 4*Input.neur_rad;
        else
            cellSize = 2*Input.neur_rad;
        end       
        if Input.neur_rad<7
            V=interp3(X,Y,Z,recon{k}(:,:,[1:Input.native_focal_plane-1 Input.native_focal_plane+1:size(recon{k},3)]),Xq,Yq,Zq);
        else
            V=recon{k};
        end
        si_V{worker}=size(V,3);
        I=zeros(size(V)+[0 0 2*opts.border(3)],'single');
        I(:,:,1+opts.border(3):si_V{worker}+opts.border(3))=single(V);
        for k=0:opts.border(3)-1
            I(:,:,opts.border(3)-k)=I(:,:,opts.border(3)+1-k)*0.96;
            I(:,:,opts.border(3)+si_V{worker}+k)=I(:,:,opts.border(3)+si_V{worker}-1+k)*0.96;
        end
        img{worker}=full(I/max(I(:)));
        siz_I{worker}=size(I);
    end

    parfor worker=1:min(opts.NumWorkers,size(recon,2)-(kk-1))
        filtered_Image_=band_pass_filter(img{worker}, cellSize, 8, gimp(mod(worker-1,n)+1),1.2);
        segm_{worker}=filtered_Image_(opts.border(1):size(filtered_Image_,1)-opts.border(1),opts.border(2):size(filtered_Image_,2)-opts.border(2),opts.border(3)+1:opts.border(3)+si_V{worker});
        if gpu
            gpuDevice([]);
        end
    end
    for kp=1:min(opts.NumWorkers,size(recon,2)-(kk-1))
        filtered_Image=zeros(siz_I{kp}-[0 0 2*opts.border(3)]);
        filtered_Image(opts.border(1):siz_I{kp}(1)-opts.border(1),opts.border(2):siz_I{kp}(2)-opts.border(2),:)=segm_{kp};
        if Input.neur_rad<7
            segmm{kk+kp-1}=filtered_Image(1:2:end,1:2:end,:);
        else
            segmm{kk+kp-1}=filtered_Image;
        end        
    end
    disp(kk)
end

for ix=1:size(recon,2)
    Vol = recon{ix}*0;
    Vol(opts.border(3):end-opts.border(3),opts.border(3):end-opts.border(3),:) = segmm{ix}(opts.border(3):end-opts.border(3),opts.border(3):end-opts.border(3),:);
    segmm{ix} = Vol;
end

end