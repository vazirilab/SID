function forward_model=generate_LFM_library_GPU(recon,centers,neur_id,psf,options)
% GENERATE_LFM_LIBRARY_GPU: algorithm computes for each detected neuron the
% LFM-image of a volume, of the appropriate size, that is empty, except for
% a small volume around the location of the neuron where it is the mean of
% this sub-volume of the components of 'recon' where the neuron was found
% according to 'neur_id'.
%
% Input:
% recon...              M cell-array of reconstructed LFM Volumes.
% centers...            N x 3 array of neuronal centers.
% neur_id...            N x M binary array; true if and only if neuron J 
%                       was found in cell K of array 'recon.                   
% psf...                LFM-point-spread-function.
% struct options
% options.gpu_ids       ids of GPUs available to the algorithm
% options.NumWorkers    Number of Workers to run the individual jobs on the
%                       GPUs in parallel.
% options.ker_shape     see reconstruct_S help
% options.ker_param     see reconstruct_S help
%
% Output:
% forward_model...      N x number of pixels array; representing the 
%                       linearized LFM image of each neuron.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('generating library');
poolobj = gcp('nocreate');
if isempty(poolobj)||(poolobj.NumWorkers~=options.NumWorkers)
    delete(poolobj);
    poolobj=parpool(options.NumWorkers);
end

if nargin<5
    options=[];
    options.NumWorkers=12;
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
    disp(['Generation of library entry ' num2str(kk) ' to ' ...
        num2str(min(kk+poolobj.NumWorkers-1,size(centers,1))) ' completed']);
end

end