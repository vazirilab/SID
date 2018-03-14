function forward_model=generate_LFM_library_GPU(recon,centers,neur_id,psf,options)


disp('generating library');
poolobj = gcp('nocreate');
delete(poolobj);

if nargin<5
    options=[];
    options.NumWorkers=12;
    options.p=2;
    options.maxIter=3;
    options.mode='basic';
    options.gpu_ids=[1 2 4 5];
end

if ~isfield(options,'NumWorkers')
    if isfield(options,'gpu_ids')
        options.NumWorkers=length(options.gpu_ids);
    else
        options.NumWorkers=1;
    end
end

if ~isfield(options,'gpu_ids')
    options.gpu_ids=[1,2,4];
end

if ~isfield(options,'ker_shape')
    options.ker_shape = 'ball';
end

if isfield(options,'ker_param')
    kernel = generate_kernel(options.ker_shape,options.ker_param);
else
    kernel = [];
end

gimp=options.gpu_ids;
parpool(options.NumWorkers);
forward_model=zeros(size(centers,1),prod(options.image_size));

for kk=1:options.NumWorkers:size(centers,1)
    Volume=cell(options.NumWorkers,1);
    H=cell(options.NumWorkers,1);
    frwd=cell(options.NumWorkers,1);
    for worker=1:min(options.NumWorkers,size(centers,1)-(kk-1))
        x=round(centers(kk+worker-1,1));
        y=round(centers(kk+worker-1,2));
        z=round(centers(kk+worker-1,3));
        Volume{worker}=zeros([options.image_size min(z+ceil(options.neur_rad/options.axial),...
            size(psf.H,5))-max(z-ceil(options.neur_rad/options.axial),1)+1]);
        
        ids=neur_id(kk+worker-1,1:size(neur_id,2));
        ids=find(ids);
        for k=1:length(ids)
            Volume{worker}(max(x-options.neur_rad,1):min(x+options.neur_rad,options.image_size(1)),...
                max(y-options.neur_rad,1):min(y+options.neur_rad,options.image_size(2)),:)=...
                Volume{worker}(max(x-options.neur_rad,1):min(x+options.neur_rad,options.image_size(1)),...
                max(y-options.neur_rad,1):min(y+options.neur_rad,options.image_size(2)),:)+...
                recon{ids(k)}(max(x-options.neur_rad,1):min(x+options.neur_rad,options.image_size(1)),...
                max(y-options.neur_rad,1):min(y+options.neur_rad,options.image_size(2)),...
                max(z-ceil(options.neur_rad/options.axial),1):min(z+ceil(options.neur_rad/options.axial),size(psf.H,5)));
            
        end
        Volume{worker}=Volume{worker}/length(ids);
        
        H{worker}=psf.H(:,:,:,:,max(z-ceil(options.neur_rad/options.axial),1):min(z+...
            ceil(options.neur_rad/options.axial),size(psf.H,5)));
    end
    
    parfor worker=1:min(options.NumWorkers,size(centers,1)-(kk-1))
        tic
        gpuDevice(gimp(mod((worker-1),length(gimp))+1));
        kern = gpuArray(kernel);
        forwardFUN_raw = @(Xguess) forwardProjectGPU(H{worker}, Xguess );
        if isempty(kernel)
            forwardFUN = forwardFUN_raw;
        else
            forwardFUN = @(Xguess) forwardFUN_raw(convn(Xguess,kern,'same'));
        end
        frwd{worker}=gather(forwardFUN(gpuArray(Volume{worker})));
        toc
        reset(gpuDevice);
        gpuDevice([]);
    end
    
    for kp=1:min(options.NumWorkers,size(centers,1)-(kk-1))
        forward_model(kk+kp-1,:)=frwd{kp}(:);
    end
    disp(kk)
end

end