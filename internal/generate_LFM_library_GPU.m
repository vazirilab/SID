function forward_model=generate_LFM_library_GPU(recon,centers,neur_id,std_image,Input,psf,options)


disp('generating library');
poolobj = gcp('nocreate');
delete(poolobj);

if nargin<7
    options=[];
    options.num_Workers=8;
    options.p=2;
    options.maxIter=3;
    options.mode='basic';
    options.gpu_ids=[1 2 4 5];
end

if ~isfield(options,'num_Workers')
    options.num_Workers=6;
end

if ~isfield(options,'gpu_ids')
    options.gpu_ids=[1,2,4];
end


gimp=options.gpu_ids;
parpool(options.num_Workers);
forward_model=zeros(size(centers,1),length(std_image(:)));

for kk=1:options.num_Workers:size(centers,1)
    Volume=cell(options.num_Workers,1);
    H=cell(options.num_Workers,1);
    frwd=cell(options.num_Workers,1);
    for worker=1:min(options.num_Workers,size(centers,1)-(kk-1))
        x=round(centers(kk+worker-1,1));
        y=round(centers(kk+worker-1,2));
        z=round(centers(kk+worker-1,3));
        Volume{worker}=zeros([size(std_image) min(z+ceil(Input.thres/Input.axial),...
            size(psf.H,5))-max(z-ceil(Input.thres/Input.axial),1)+1]);
        
        ids=neur_id(kk+worker-1,1:size(neur_id,2));
        ids=find(ids);
        for k=1:length(ids)
            Volume{worker}(max(x-Input.thres,1):min(x+Input.thres,size(std_image,1)),...
                max(y-Input.thres,1):min(y+Input.thres,size(std_image,2)),:)=...
                Volume{worker}(max(x-Input.thres,1):min(x+Input.thres,size(std_image,1)),...
                max(y-Input.thres,1):min(y+Input.thres,size(std_image,2)),:)+...
                recon{ids(k)}(max(x-Input.thres,1):min(x+Input.thres,size(std_image,1)),...
                max(y-Input.thres,1):min(y+Input.thres,size(std_image,2)),...
                max(z-ceil(Input.thres/Input.axial),1):min(z+ceil(Input.thres/Input.axial),size(psf.H,5)));
                
        end
        Volume{worker}=Volume{worker}/length(ids);
       
        H{worker}=psf.H(:,:,:,:,max(z-ceil(Input.thres/Input.axial),1):min(z+...
            ceil(Input.thres/Input.axial),size(psf.H,5)));   
    end
    
    parfor worker=1:min(options.num_Workers,size(centers,1)-(kk-1))
        tic
        gpuDevice(gimp(mod((worker-1),length(gimp))+1));
        frwd{worker}=gather(forwardProjectGPU(gpuArray(H{worker}), gpuArray(Volume{worker})));
        toc
        gpuDevice([]);
    end
    
    for kp=1:min(options.num_Workers,size(centers,1)-(kk-1))
        forward_model(kk+kp-1,:)=frwd{kp}(:);
    end
    disp(kk)
end

end