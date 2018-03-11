function recon_nnmf=reconstruct_S(S,psf_ballistic,opts)
% RECONSTRUCT_S: algorithm performs LFM reconstruction with a kernel of the
% component of S with optional help of GPUs.
%
%       LFM-image ~ psf_ballistic.H*conv(recon_nnmf,kernel,'same')
%
% Input:
% S...              Vector of LFM images
% struct opts
%   opts.shape      shape of the kernel 
%       'gaussian'  gaussian kernel
%       'ball'      binary kernel in shape of ball
%       'lorentz'   laurentzian kernel
%       'user'      general kernel defined by user
%   opts.ker_param  'string', Parameter for the kernel, the following possibilities:
%       'gaussian'  2-vector; First component is standard deviation in
%                   lateral directions, Second component is standard
%                   deviation in the axial direction.
%       'ball'      2-vector: First component is the radius in the lateral
%                   direction, second component is the radius in the axial
%                   direction
%       'lorentz'   2-vector (accordingly)
%       'user'      3D-kernel
%   maxIter         Number of terations of the deconvolution algorithm
%   whichSolver     String; Algorithm for deconvolution. Options are
%                   'ISRA' for the Image space reconstruction algorithm
%                   implemented by deconvRL, or 'fast_nnls' for gradient
%                   descent with exact line search, implemented by
%                   fast_nnls.
%   gpu_ids         Vector of ids of gpu's, that are available. If this
%                   array is empty the algorithm will perform reconstruction 
%                   on the CPU
%   NumWorkers      number of workers for parpool
%   p, mode,lambda,
%   lambda_         See help of deconv-algorithm
% q...              Quantile of the image, where it should be thresholded.
% 
% Output:
%   recon_nnmf...   cell-array of reconstructed Volumes

if nargin<3
    opts = struct;
end

if ~isfield(opts,'NumWorkers')
    if isfield(opts,'gpu_ids')
        opts.NumWorkers=length(opts.gpu_ids);
    else
        opts.NumWorkers=1;
    end
end

if ~isfield(opts,'gpu_ids')
    opts.gpu_ids=[];
end

if ~isfield(opts,'whichSolver')
    opts.whichSolver = 'fast_nnls';
end

if ~isfield(opts,'ker_shape')
    opts.ker_shape = 'ball';
end

if ~isfield(opts,'maxIter')
    opts.maxIter = 8;
end

if ~isfield(opts,'q')
    opts.q = 0.8;
end

if isfield(opts,'gpu_ids')
    n = length(opts.gpu_ids);
    gimp = opts.gpu_ids;
else
    n = 1;
    gimp = -1;
end

disp('Allocating memory');
recon_nnmf=cell(size(S,1),1);
for l=1:size(S,1)
    recon_nnmf{l} = ones([size(S,2) size(S,3) size(psf_ballistic.H,5)],'single');
end

disp('Generating kernel');

if isfield(opts,'ker_param')
    if strcmp(opts.ker_shape,'ball')
        [X,Y,Z] = meshgrid(-opts.ker_param(1):opts.ker_param(1),-...
            opts.ker_param(1):opts.ker_param(1),-opts.ker_param(2):opts.ker_param(2));
        kernel = single((X.^2/opts.ker_param(1)^2 + Y.^2/opts.ker_param(1)^2 +...
            Z.^2/opts.ker_param(2)^2)<=1);
    elseif strcmp(opts.ker_shape,'gaussian')
        gaussian=fspecial('gaussian',ceil(10*opts.ker_param(1))+1,...
            opts.ker_param(1));
        kernel=reshape(reshape(gaussian,[],1)*exp(-[-3*opts.ker_param(2):...
            1:3*opts.ker_param(2)].^2/4/opts.ker_param(2)^2),ceil(...
            10*opts.ker_param(1))+1,ceil(10*opts.ker_param(1))+1,[]);
    elseif strcmp(opts.ker_shape,'lorentz')
        [X,Y,Z] = meshgrid([-ceil(5*opts.ker_param(1)):ceil(...
            5*opts.ker_param(1))],[-ceil(5*opts.ker_param(1)):...
            ceil(5*opts.ker_param(1))],[-ceil(5*opts.ker_param(2)):...
            ceil(5*opts.ker_param(2))]);
        kernel = 1./(1 + (opts.ker_param(1)*(X.^2 + Y.^2) + opts.ker_param(2)*Z.^2));
    elseif strcmp(opts.ker_shape,'user')
        kernel=opts.ker_param;
    end
    kernel = kernel/norm(kernel(:));
else
    kernel = [];
end
disp('Prepared Reconstruction kernel');

if isempty(opts.gpu_ids)
    disp('No GPUs selected -> CPU reconstruction');
    forwardFUN_raw =  @(Volume) forwardProjectACC( psf_ballistic.H, Volume, ...
        psf_ballistic.CAindex );
    backwardFUN_raw = @(projection) backwardProjectACC_new(psf_ballistic.H, ...
        projection, psf_ballistic.CAindex);
    if isempty(kernel)
        forwardFUN =  backwardFUN_raw;
        backwardFUN = forwardFUN_raw;
    else
        forwardFUN = @(Xguess) forwardFUN_raw(convn(Xguess,kernel,'same'));
        backwardFUN = @(projection) convn(backwardFUN_raw(projection),kernel,'same');
    end
    
    for ii = 1:size(S,1)
        img = S(ii,:,:);
        img = max(img - quantile(img(opts.microlenses(:)==0),opts.q),0);
        img = img/max(img(:));
        disp('Backprojecting...');
        Htf = backwardFUN(single(img));
        disp('Backprojection completed');
        disp('Reconstructing...')
        recon_nnmf{ii} = fast_deconv(forwardFUN, backwardFUN, Htf, opts.maxIter, opts);
        disp(['Reconstruction of frame ' num2str(ii) ' completed']);
    end
else
    disp('GPU reconstruction')
    poolobj = gcp('nocreate');
    H=psf_ballistic.H;
    if isempty(poolobj)||(poolobj.NumWorkers~=opts.NumWorkers)
        delete(poolobj);
        poolobj=parpool(opts.NumWorkers);
    end
    disp([num2str(poolobj.NumWorkers) ' Workers reconstruct on ' ...
        num2str(length(opts.gpu_ids)) ' GPUs']);
    for kk=1:opts.NumWorkers:size(S,1)
        img_cell=cell(opts.NumWorkers,1);
        options=cell(opts.NumWorkers,1);
        for worker=1:min(opts.NumWorkers,size(S,1)-(kk-1))
            disp(['Preparing job: ' num2str(worker)]);
            ii=kk+worker-1;
            img = S(ii,:,:);
            img = max(img - quantile(img(opts.microlenses(:)==0),opts.q),0);
            img = img/max(img(:));
            img_cell{worker}=img;
            options{worker}=opts;
            options{worker}.gpu_id=gimp(mod(worker-1,n)+1);
        end
        
        parfor worker=1:min(opts.NumWorkers,size(S,1)-(kk-1))
            gpuDevice(options{worker}.gpu_id);
            disp([datestr(now, 'YYYY-mm-dd HH:MM:SS')...
                ': Starting batch reconstrution in worker ' num2str(worker)...
                ' on GPU ' num2str(options{worker}.gpu_id)]);
            kern = gpuArray(kernel);
            backwardFUN_raw = @(projection) backwardProjectGPU_new(H, projection );
            forwardFUN_raw = @(Xguess) forwardProjectGPU(H, Xguess );
            if isempty(kernel)
                forwardFUN = forwardFUN_raw;
                backwardFUN = backwardFUN_raw;
            else
                forwardFUN = @(Xguess) forwardFUN_raw(convn(Xguess,kern,'same'));
                backwardFUN = @(projection) convn(backwardFUN_raw(projection),kern,'same');
            end
            Xguess = backwardFUN(img_cell{worker});
            if strcmp(options{worker}.whichSolver,'ISRA')
                Xguess = deconvRL(forwardFUN, backwardFUN, Xguess, ...
                    options{worker}.maxIter, Xguess ,options{worker});
            elseif strcmp(options{worker}.whichSolver,'fast_nnls')
                Xguess=fast_deconv(forwardFUN, backwardFUN, Xguess, ...
                    options{worker}.maxIter,options{worker});
            elseif strcmp(options{worker}.whichSolver,'fast_ls')
                Xguess=fast_deconv_neg(forwardFUN, backwardFUN, Xguess, ...
                    options{worker}.maxIter,options{worker});
            end
            recon{worker} = gather(Xguess);
        end
        for kp=1:min(opts.NumWorkers,size(S,1)-(kk-1))
            recon_nnmf{kk+kp-1}=recon{kp};
        end
        disp(['Reconstruction of frames from ' num2str(kk) ' to ' ...
            num2str(kk+poolobj.NumWorkers-1) ' completed']);
    end
end

disp('Reconstruction of S completed');
end
