function Xguess = reconstruction_gpu_oliver(in_file, psf_file, tmp_dir, maxIter, options,opts)

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

disp('Iteration method: ISRA');

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
Htf = backwardFUN(LFIMG);
ttime = toc;
disp(['  iter ' num2str(0) ' | ' num2str(maxIter) ', took ' num2str(ttime) ' secs']);
Xguess = Htf;

%% write Htf to tmp file in order to avoid serialization for inter-process-communication to batch jobs
Htf_filename = fullfile(tmp_dir, 'Htf_tmp.mat');
save(Htf_filename, 'Htf');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xguess = deconvRL(forwardFUN, backwardFUN, Htf, maxIter, Xguess ,opts);

delete(Htf_filename);
Xguess=gather(Xguess);
disp(['Volume reconstruction complete.']);
end
