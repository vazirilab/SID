function Xguess = reconstruction_gpu_oliver(in_file, psf_file, save_dir, tmp_dir, whichSolver, maxIter, options,opts)

if nargin < 7
    options = struct;
end

if nargin < 6
    maxIter = 8;
end

if nargin < 5
    whichSolver = 1;
end

if ~isfield(options, 'gpu_ids')
    options.gpu_ids = [4 5]; %the GPU ids to use for this job (valid on pcl-imp-2: 1,2,4,5)
end

warning('off');

if not(isdeployed)
    addpath('./Util/');
    addpath('./Solver/');
end
%%
eqtol = 1e-10;
GPUcompute = 1;
cluster = parcluster('local')
edgeSuppress = 0;

if whichSolver==1, %% Richardson-Lucy : deconvolution
    disp('Iteration method: ISRA');
elseif whichSolver==2, %% PCG
    disp('Iteration method: Preconditioned Congugate Gradient');
elseif whichSolver==3, %% SART : solver for tomography
    disp('Iteration method: Simultaneous Algebraic Reconstruction Technique');
elseif whichSolver==4, %% ADMM : with sparsity constraint
    disp('Iteration method: Alternating-Direction Method of Multipliers with sparsity constraints');
elseif whichSolver==5, %% YALL1 : compressive sensing
    disp('Iteration method: Your ALgorithm for L1');
elseif whichSolver==6, %% Richardson-Lucy with ground truth
    disp('Iteration method: Richardson-Lucy with ground truth. WARNING: EXPERIMENTAL!');
elseif whichSolver==7, %% Richardson-Lucy with ground truth
    disp('Iteration method: Richardson-Lucy with convolved-deconvolved ground truth. WARNING: EXPERIMENTAL!');
end

if ~(exist(save_dir)==7)
    mkdir(save_dir);
end
if ~(exist(tmp_dir)==7)
    mkdir(tmp_dir);
end

%% REPARE PARALLEL COMPUTING
pool = gcp('nocreate')
if ~pool.isvalid % checking to see if my pool is already open
    pool = parpool(cluster)
end

%% Load Data
% load(psf_file);
if strcmp(class(psf_file.H), 'double')
    psf_file.H = single(psf_file.H);
    psf_file.Ht = single(psf_file.Ht);
end
% disp(['Successfully loaded PSF matrix : ' psf_file]);
disp(['Size of PSF matrix is : ' num2str(size(psf_file.H)) ]);

if ~isa(in_file, 'struct')
    in_matfile = matfile(in_file);
    [n_px_x, n_px_y, n_frames] = size(in_matfile, 'LFmovie');
else
    in_matfile = in_file;
    [n_px_x, n_px_y, n_frames] = size(in_file.LFmovie);
end


ReconFrames = 1:n_frames;
lightFieldResolution = [n_px_x n_px_y];
global volumeResolution;
volumeResolution = [n_px_x n_px_y size(psf_file.H,5)];
disp(['Image size is ' num2str(volumeResolution(1)) 'X' num2str(volumeResolution(2))]);

%% prepare reconstruction of first frame
if GPUcompute,
    Nnum = size(psf_file.H,3);
    backwardFUN = @(projection) backwardProjectGPU(psf_file.Ht, projection );
    forwardFUN = @(Xguess) forwardProjectGPU( psf_file.H, Xguess );
    %prepare global gpu variables for reconstruction of the first frame.
    %this does not affect the parallelization later on.
    gpuDevice(options.gpu_ids(1));
    global zeroImageEx;
    global exsize;
    xsize = [volumeResolution(1), volumeResolution(2)];
    msize = [size(psf_file.H,1), size(psf_file.H,2)];
    mmid = floor(msize/2);
    exsize = xsize + mmid;
    exsize = [ min( 2^ceil(log2(exsize(1))), 256*ceil(exsize(1)/256) ), min( 2^ceil(log2(exsize(2))), 256*ceil(exsize(2)/256) ) ];
    zeroImageEx = gpuArray(zeros(exsize, 'single'));
    disp(['FFT size is ' num2str(exsize(1)) 'X' num2str(exsize(2))]);
else
    forwardFUN =  @(Xguess) forwardProjectACC( psf_file.H, Xguess, CAindex );
    backwardFUN = @(projection) backwardProjectACC(psf_file.Ht, projection, CAindex );
end

%% Reconstruct First Frame To Get Normalisation Constant
k = 1;
frame = ReconFrames(k);
LFIMG = single(in_matfile.LFmovie(:,:,frame));
tic; 
% if p==-1
%     Htf = backwardFUN(del2(LFIMG)); 
% 
% else
Htf = backwardFUN(LFIMG); 
% end
ttime = toc;
disp(['  iter ' num2str(0) ' | ' num2str(maxIter) ', took ' num2str(ttime) ' secs']);
Xguess = Htf;

%% write Htf to tmp file in order to avoid serialization for inter-process-communication to batch jobs
Htf_filename = fullfile(tmp_dir, 'Htf_tmp.mat');
save(Htf_filename, 'Htf');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if whichSolver==1, %% Richardson-Lucy : deconvolution

    Xguess = deconvRL(forwardFUN, backwardFUN, Htf, maxIter, Xguess ,opts);
    
elseif whichSolver==2, %% PCG
    forwardbackwardFUN_vec = @(Xguess) reshape(backwardFUN(forwardFUN(Xguess)), [prod(volumeResolution) 1 1]);
    [Xguess, flag] = pcg(forwardbackwardFUN_vec,Htf(:),eqtol,maxIter);
    disp(flag);
    Xguess(Xguess<0) = 0;
    Xguess = reshape(Xguess, volumeResolution);
    
elseif whichSolver==3, %% SART : solver for tomography
    Xguess = SART(forwardFUN, backwardFUN, LFIMG, [], [], Htf, maxIter, false);
    
elseif whichSolver==4, %% ADMM : with sparsity constraint
    forwardFUN_vec = @(Xguess) reshape( forwardFUN( reshape(Xguess,volumeResolution) ), [prod(lightFieldResolution(:)) 1]);
    backwardFUN_vec = @(projection) reshape(backwardFUN( reshape(projection,lightFieldResolution) ), [prod(volumeResolution) 1]);
    lambda = 0.0001;
% lambda=0.0003*0.8688;
% lambda=0;
    alpha   = 1.0;
    rho     = 1.0;
    maxIterSART = 7;
    [Xguess, FitInfo] = lasso(forwardFUN_vec, backwardFUN_vec, prod(lightFieldResolution), prod(volumeResolution), LFIMG(:), lambda, rho, alpha, maxIter, maxIterSART, false, volumeResolution);
    disp(FitInfo);
    Xguess = reshape(Xguess, volumeResolution);
    
elseif whichSolver==5, %% YALL1 : compressive sensing
    forwardFUN_vec = @(Xguess) reshape( forwardFUN( reshape(Xguess,volumeResolution) ), [prod(lightFieldResolution(:)) 1]);
    backwardFUN_vec = @(projection) reshape(backwardFUN( reshape(projection,lightFieldResolution) ), [prod(volumeResolution) 1]);
    %         NLfun = @(Xguess) double(forwardFUN_vec(Xguess) - LFIMG(:));
    %         options.Maxiter = maxIter;
    %         Xguess = fsolve(NLfun,double(Htf(:)),options);
    A.times = forwardFUN_vec;
    A.trans = backwardFUN_vec;
    lambda = 0.01;
    opts.tol = 1e-14;
    opts.rho = lambda/2;
    opts.nonneg = 1;
    opts.maxit = maxIter;
    [Xguess, Out] = yall1(A, reshape(LFIMG, [prod(lightFieldResolution(:)) 1]), opts);
    disp(Out);
    Xguess = reshape(Xguess, volumeResolution);
elseif whichSolver==6,
    truth = squeeze(in_matfile.truth(:,:,:,1));
    [Xguess, X_intermediate_guesses, errorBack_sav] = deconvRL_truth_restricted(forwardFUN, backwardFUN, Htf, maxIter, truth, truth);    
%     [Xguess, X_intermediate_guesses, errorBack_sav] = deconvRL_truth_restricted(forwardFUN, backwardFUN, Htf, maxIter, Xguess, truth);    
    %save(fullfile(save_dir, 'intermediate_vals.mat'), ['X_intermediate_guesses' 'errorBack_sav']);
elseif whichSolver==7
    truth = squeeze(in_matfile.truth(:,:,:,1));
    [Xguess, X_intermediate_guesses, errorBack_sav] = deconvRL_LFMtruth_restricted(forwardFUN, backwardFUN, Htf, maxIter, Xguess, truth);   
else
    disp('Numerical solver is not properly selected');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% renorm_first_filename = fullfile(save_dir, 'renorm_first_frame.mat');
% if exist(renorm_first_filename)
%     load(renorm_first_filename);
% else
%     MV3Dgain = gather(0.66/max(Xguess(:)));
%     save(renorm_first_filename,'MV3Dgain')
% end
% 
% if GPUcompute,
%     XguessCPU = gather(Xguess);
% else
%     XguessCPU = Xguess;
% end
% 
% 
% Xvolume = uint16(round(65535*MV3Dgain*XguessCPU));
% if edgeSuppress,
%     Xvolume( (1:1*Nnusm), :,:) = 0;
%     Xvolume( (end-1*Nnum+1:end), :,:) = 0;
%     Xvolume( :,(1:1*Nnum), :) = 0;
%     Xvolume( :,(end-1*Nnum+1:end), :) = 0;
% end
% tic;
% if isfield(in_matfile, 'image_info_struct')
%     image_info_struct = in_matfile.image_info_struct(frame, 1);
%     name = image_info_struct.name;
% else
%     name = ['img_' num2str(frame, '%05i') '.tif'];
% end
% save_fn = fullfile(save_dir, name);
% imwrite(Xvolume(:,:,1), save_fn)
% for k = 2:size(Xvolume,3)
%     imwrite(Xvolume(:,:,k), save_fn, 'WriteMode', 'append');
% end
% ttime = toc;
% disp(['Writing time: ' num2str(ttime) ' secs']);
% %%
% clear H Htf Ht LFmovie
% delete(pool)
% 
% %% Rest of frames, using multiple GPUs
% n = length(options.gpu_ids);
% recon_handle=@batchGPU;
% 
% %build list of frames that each worker has to process
% frames_worker = cell(n, 1);
% i = 1;
% for f=2:length(ReconFrames)
%     frames_worker{i}(end+1) = ReconFrames(f);
%     i = i+1;
%     if i == n+1
%         i = 1;
%     end
% end
% disp(frames_worker)
% 
% %% build cell array of batch jobs, one for each frame, and submit them
% job_array=cell(n,2);
% for i=[1:n]
%     disp(i)
%     tmp_frames = frames_worker{i};
%     if length(tmp_frames)==0
%         continue;
%     end
%     [fp, fn, fe] = fileparts(mfilename('fullpath'));
%     fp = '/home/tobias.noebauer/devel/lfrecon_vsc'
%     tic;
%     job_array{i,2}=batch( cluster, recon_handle, 0, ...
%         {in_file, ...
%         tmp_frames, options.gpu_ids(i), psf_file, Htf_filename, ...
%         whichSolver, MV3Dgain, volumeResolution, ...
%         maxIter, edgeSuppress, Nnum, save_dir}, ...
%         'AdditionalPaths', ...
%         {fp, ...
%          fullfile(fp, '/Util/'), ...
%          fullfile(fp, '/Solver/')});
%     toc
%     job_array{i,1}=i;
% end
% 
% %% poll queue status until all jobs are done. provide some info on progress
% finished=0;
% i = 0;
% while ~finished
%     jobs = horzcat(job_array{:,2});
%     finished_jobs = 0;
%     failed_jobs = 0;
%     for k=1:length(jobs)
%         finished_jobs = finished_jobs + strcmp(jobs(k).State, 'finished');
%         failed_jobs = failed_jobs + strcmp(jobs(k).State, 'failed');
%     end
%     finished = ((finished_jobs+failed_jobs)==length(jobs));
%     
%     disp(['Time elapsed: ' num2str(i*10) ' sec. Submitted: ' num2str(length(jobs)) ' Finished: ' num2str(finished_jobs) ' Failed: ' num2str(failed_jobs)])
%     pause(10)
%     i = i+ 1;
% end
% 
% %TODO: rewrite batchGPU() to produce TIFFs. start tiffs_live_to_hdf5.py to
% %collect them

delete(Htf_filename);
Xguess=gather(Xguess);
disp(['Volume reconstruction complete.']);
end
