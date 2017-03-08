function Xguess = reconstruction_sparse(in_file, psf_ballistic, options)

if nargin < 3
    options = struct;
end

if ~isfield(options,'maxIter')
    options.maxIter = 8;
end

if ~isfield(options,'whichSolver')
    options.whichSolver = 'fast_nnls';
end

if ~isfield(options, 'gpu_ids')
    options.gpu_ids = [4 5]; %the GPU ids to use for this job (valid on pcl-imp-2: 1,2,4,5)
end

if ~isfield(options,'rad')
    options.rad = [2,2]; % radius of kernel [lateral, axial] with which volume gets convolved during reconstruction (akin to minimum expected feature size)
end

eqtol = 1e-10;
cluster = parcluster('local');
edgeSuppress = 0;

%% REPARE PARALLEL COMPUTING
pool = gcp('nocreate')
if ~pool.isvalid % checking to see if my pool is already open
    pool = parpool(cluster)
end

%% Load Data
if strcmp(class(psf_ballistic.H), 'double')
    psf_ballistic.H = single(psf_ballistic.H);
    psf_ballistic.Ht = single(psf_ballistic.Ht);
end
disp(['Size of PSF matrix is : ' num2str(size(psf_ballistic.H)) ]);


[n_px_x, n_px_y, n_frames] = size(in_file.LFmovie);

global volumeResolution;
volumeResolution = [n_px_x n_px_y size(psf_ballistic.H,5)];
disp(['Image size is ' num2str(volumeResolution(1)) 'X' num2str(volumeResolution(2))]);

%% prepare reconstruction of first frame
Nnum = size(psf_ballistic.H,3);
backwardFUN_ = @(projection) backwardProjectGPU_new(psf_ballistic.H, projection );
forwardFUN_ = @(Xguess) forwardProjectGPU( psf_ballistic.H, Xguess );
%prepare global gpu variables for reconstruction of the first frame.
%this does not affect the parallelization later on.
gpuDevice(options.gpu_ids(1));
global zeroImageEx;
global exsize;
xsize = [volumeResolution(1), volumeResolution(2)];
msize = [size(psf_ballistic.H,1), size(psf_ballistic.H,2)];
mmid = floor(msize/2);
exsize = xsize + mmid;
exsize = [ min( 2^ceil(log2(exsize(1))), 256*ceil(exsize(1)/256) ), min( 2^ceil(log2(exsize(2))), 256*ceil(exsize(2)/256) ) ];
zeroImageEx = gpuArray(zeros(exsize, 'single'));
disp(['FFT size is ' num2str(exsize(1)) 'X' num2str(exsize(2))]);

%% generate kernel
<<<<<<< HEAD
if ~isfield(options,'form')
    options.form='spherical';
end

if isfield(options,'rad')
    if prod(options.rad)>0
        if strcmp(options.form,'spherical')
            W=zeros(2*options.rad(1)+1,2*options.rad(1)+1,2*options.rad(2)+1);
            for ii=1:2*options.rad(1)+1
                for jj=1:2*options.rad(1)+1
                    for kk=1:2*options.rad(2)+1
                        if  ((ii-(options.rad(1)+1))^2/options.rad(1)^2+(jj-(options.rad(1)+1))^2/options.rad(1)^2+(kk-(options.rad(2)+1))^2/options.rad(2)^2)<=1
                            W(ii,jj,kk)=1;
                        end
                    end
                end
=======
W=zeros(2*options.rad(1)+1,2*options.rad(1)+1,2*options.rad(2)+1);
for ii=1:2*options.rad(1)+1
    for jj=1:2*options.rad(1)+1
        for kk=1:2*options.rad(2)+1
            if  ((ii-(options.rad(1)+1))^2/options.rad(1)^2+(jj-(options.rad(1)+1))^2/options.rad(1)^2+(kk-(options.rad(2)+1))^2/options.rad(2)^2)<=1
                W(ii,jj,kk)=1;
>>>>>>> refs/remotes/origin/master
            end
        elseif strcmp(options.form,'gaussian')
            gaussian=fspecial('gaussian',ceil(10*options.rad(1))+1,options.rad(1));
            W=reshape(reshape(gaussian,[],1)*exp(-[-3*options.rad(2):1:3*options.rad(2)].^2/4/options.rad(2)^2),ceil(10*options.rad(1))+1,ceil(10*options.rad(1))+1,[]);
            
        elseif strcmp(options.form,'lorentz')
            [X Y Z] = meshgrid([-ceil(5*options.rad(1)):ceil(5*options.rad(1))],[-ceil(5*options.rad(1)):ceil(5*options.rad(1))],[-ceil(5*options.rad(2)):ceil(5*options.rad(2))]);
            W = 1./(1 + (options.rad(1)*(X.^2 + Y.^2) + options.rad(2)*Z.^2));
        end
        W = W/norm(W(:));
        kernel=gpuArray(W);
        
        forwardFUN = @(Xguess) forwardFUN_(convn(Xguess,kernel,'same'));
        backwardFUN = @(projection) convn(backwardFUN_(projection),kernel,'same');
    else
        forwardFUN = forwardFUN_;
        backwardFUN = backwardFUN_;
    end
    
else
    forwardFUN = forwardFUN_;
    backwardFUN = backwardFUN_;
end
<<<<<<< HEAD
=======
kernel=gpuArray(W);

forwardFUN = @(Xguess) forwardFUN_(convn(Xguess, kernel, 'same'));
backwardFUN = @(projection) convn(backwardFUN_(projection), kernel, 'same');
>>>>>>> refs/remotes/origin/master

%% Reconstruction
LFIMG = single(in_file.LFmovie);
tic;
Xguess = backwardFUN(LFIMG);
ttime = toc;
disp(['  iter ' num2str(0) ' | ' num2str(options.maxIter) ', took ' num2str(ttime) ' secs']);

%%
if strcmp(options.whichSolver,'ISRA')
    Xguess = deconvRL(forwardFUN, backwardFUN, Xguess, options.maxIter, Xguess ,options);
elseif strcmp(options.whichSolver,'fast_nnls')
    Xguess=fast_deconv(forwardFUN, backwardFUN, Xguess, options.maxIter,options);
end

Xguess=gather(Xguess);
disp('Volume reconstruction complete.');
end
