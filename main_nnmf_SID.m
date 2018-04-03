function main_nnmf_SID(indir, outdir, psffile, x_offset, y_offset, dx, optional_args)
%% fish timeseries extraction
% all Input fields used in the rest of the script:
% Input.LFM_folder
% Input.psf_filename_ballistic
% Input.output_folder
% Input.x_offset
% Input.y_offset
% Input.dx

%%% DETRENDING
% Input.detrend -- boolean, whether to perform detrending prior to NMF
% Input.delta  -- integer, Half width of sliding window (in units of frames) for
% low-pass filtering the frame means prior to detrending. Set this to a value that is large
% compared to the duration of a Ca transient (e.g. 10 times as large), to
% avoid that the detrending smoothes out true Ca transients.

%% NMF
% Input.nnmf_opts.rank
% Input.SID_output_name
% Input.tmp_dir
% Input.bg_iter
% Input.rectify
% Input.Junk_size
% Input.bg_sub
% Input.gpu_ids
% Input.num_iter
% Input.native_focal_plane
% Input.neur_rad
% Input.nnmf_opts
% Input.recon_opts
% Input.update_template
% Input.fluoslide_fn

% Input.frames.start = 1;%frames_for_model_optimization
% Input.frames.step = 10;
% Input.frames.end = 1e6;
% Input.frames.mean = 1; % boolean (true == take mean over frames,

%% Required parameters
Input.LFM_folder = indir;
Input.psf_filename_ballistic = psffile;
Input.output_folder = outdir;
Input.x_offset = x_offset;
Input.y_offset = y_offset;
Input.dx = dx;

%% Optional parameters
if isfield(optional_args, 'out_filename')
    Input.SID_output_name = optional_args.out_filename;
else
    Input.SID_output_name = ['nnmf_sid_result_' datestr(now, 'YY-mm-ddTHHMM') '.mat'];
end

if isfield(optional_args, 'tmp_dir')
    Input.tmp_dir = optional_args.tmp_dir;
else
    Input.SID_output_name = tempdir();
end


if isfield(optional_args, 'bg_iter')
    Input.bg_iter = optional_args.bg_iter;
else
    Input.bg_iter = 2;
end

if isfield(optional_args, 'rectify')
    Input.rectify = optional_args.rectify;
else
    Input.rectify = 1;
end

if isfield(optional_args, 'junk_size')
    Input.Junk_size = optional_args.junk_size;
else
    Input.Junk_size = 1000;
end

if isfield(optional_args, 'bg_sub')
    Input.bg_sub = optional_args.bg_sub;
else
    Input.bg_sub = 1;
end

if ~isfield(Input,'segmentation')
    Input.segmentation.threshold = 0.01;
end

if ~isfield(Input.segmentation,'top_cutoff')
    Input.segmentation.top_cutoff = 1;
end


if isfield(optional_args, 'frames')
    Input.frames = optional_args.frames;
else
    Input.frames.start = 1;
    Input.frames.step = 10;
    Input.frames.end = inf;
    Input.frames.mean = false;
end

if isfield(optional_args, 'optimize_kernel')
    Input.optimize_kernel = optional_args.optimize_kernel;
else
    Input.optimize_kernel = 1;
end

if isfield(optional_args, 'gpu_ids')
    Input.gpu_ids = optional_args.gpu_ids;
else
    Input.gpu_ids = [];
end

if isfield(optional_args, 'n_iter')
    Input.num_iter = optional_args.n_iter;
else
    Input.num_iter = 4;
end

if isfield(optional_args, 'native_focal_plane')
    Input.native_focal_plane = optional_args.native_focal_plane;
else
    Input.native_focal_plane = 26;
end

% typical neuron radius in px. Typically 6 for fish using 20x/0.5
% objective, 9-12 for mouse cortex and 16x/0.8
if isfield(optional_args, 'neuron_radius_px')
    Input.neur_rad = optional_args.neuron_radius_px;
else
    Input.neur_rad = 8;
end

if isfield(optional_args, 'recon_opts')
    Input.recon_opts = optional_args.recon_opts;
else
    Input.recon_opts.p=2;
    Input.recon_opts.maxIter=8;
    Input.recon_opts.mode='basic';
    Input.recon_opts.lambda_=3.5;
    Input.recon_opts.ker_shape='user';
end

if isfield(optional_args, 'filter')
    Input.filter = optional_args.filter;
else
    Input.filter = 0;
end

if isfield(optional_args, 'total_deconv_opts')
    Input.total_deconv_opts = optional_args.total_deconv_opts;
else
    Input.total_deconv_opts = [];
end

if isfield(optional_args, 'psf_cache_dir')  % a very fast storage location (ideally, a ramdisk), for caching the psf file. This is to avoid serialization to parfor workers
    Input.psf_cache_dir = optional_args.psf_cache_dir;
else
    Input.psf_cache_dir = '/dev/shm';
end

if isfield(optional_args, 'do_crop')
    Input.do_crop = optional_args.do_crop;
else
    Input.do_crop = true;
end

% Width of borders to crop, in units of microlenses. Set to empty array to
% disable. When giving a value of floor([ix1_lo_border_width ix1_hi_border_width ix2_hi_border_width
% ix2_hi_border_width] / Nnum)
% that means that
% cropped_img = full_img(ix1_lo_border_width + 1 : end - ix1_hi_border_width, ix2_lo_border_width + 1 : end - ix2_hi_border_width)
if isfield(optional_args, 'crop_border_microlenses')
    Input.crop_border_microlenses = optional_args.crop_border_microlenses;
else
    Input.crop_border_microlenses = [0 0 0 0];
end

if ~isfield(Input,'cluster_iter')
    Input.cluster_iter=40;
end
%%
Input.axial = 4;
Input.nnmf_opts.max_iter = 600;
Input.nnmf_opts.lamb_temp = 0;
Input.nnmf_opts.lamb_spat = 0;
Input.nnmf_opts.lamb_orth = 5e-4;
Input.nnmf_opts.rank = 30;
Input.update_template = false;
Input.detrend = true;
Input.optimize_kernel = false;
mkdir(Input.output_folder)

%% Cache and open PSF
if ~strcmp(Input.psf_cache_dir, '')
    [~, rand_string] = fileparts(tempname());
    Input.psf_cache_dir_unique = fullfile(Input.psf_cache_dir, ['sid_nnmf_recon_psf_' rand_string]);
    disp(['Creating tmp dir for psf caching: ' Input.psf_cache_dir_unique]);
    mkdir(Input.psf_cache_dir_unique);
    disp('Copying psf file to tmp dir for caching...');
    copyfile(Input.psf_filename_ballistic, Input.psf_cache_dir_unique);
    [~, psf_fname, psf_ext] = fileparts(Input.psf_filename_ballistic);
    Input.psf_filename_ballistic_in = Input.psf_filename_ballistic;
    Input.psf_filename_ballistic = fullfile(Input.psf_cache_dir_unique, [psf_fname psf_ext]);
    clear psf_fname;
end
psf_ballistic = matfile(Input.psf_filename_ballistic);

if ~isfield(Input.segmentation,'bottom_cutoff')
    Input.segmentation.bottom_cutoff = size(psf_ballistic.H,5);
end

%%
% Input.fluoslide_fn = ['fluoslide_Nnum' num2str(psf_ballistic.Nnum) '.mat'];
% if ~exist(Input.output_folder, 'dir')
%     mkdir(Input.output_folder);
% end

%% Prepare cluster object
pctconfig('portrange', [27400 27500] + randi(100)*100);
cluster = parcluster('local');
if ~isfield(Input, 'job_storage_location')
    Input.job_storage_location = tempdir();
end
[~, rand_string] = fileparts(tempname());
Input.job_storage_location_unique = fullfile(Input.job_storage_location, ['nnmf_sid_' rand_string]);
if ~exist(Input.job_storage_location_unique, 'dir')
    mkdir(Input.job_storage_location_unique);
end
cluster.JobStorageLocation = Input.job_storage_location_unique;
disp(cluster);
delete(gcp('nocreate'));

%% Load mask image
if isfield(Input, 'mask_file')
    Input.mask = logical(imread(Input.mask_file));
else
    Input.mask = true;
end
figure; imagesc(double(Input.mask), [0 1]); axis image; colorbar;
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_mask.pdf']), '-dpdf', '-r300');

%% Compute bg components via rank-1-factorization
% disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Computing background components']);
% if Input.bg_sub
%     [SID_output.bg_temporal, SID_output.bg_spatial] = par_rank_1_factorization(Input.LFM_folder, Input.frames.step, Input.bg_iter, 0, 0, 0, 0, Input.frames.end, Input.mask);
% else
%     SID_output.bg_temporal = [];
%     SID_output.bg_spatial = [];
% end
%
% SID_output.bg_spatial_pre_crop_rectify = SID_output.bg_spatial;
%
% if Input.rectify
%     SID_output.bg_spatial = ImageRect(SID_output.bg_spatial_pre_crop_rectify, Input.x_offset, Input.y_offset, Input.dx, psf_ballistic.Nnum, ...
%         true, Input.crop_border_microlenses(3), Input.crop_border_microlenses(4), Input.crop_border_microlenses(1), Input.crop_border_microlenses(2));
% else
%     Nnum = psf_ballistic.Nnum;
%     SID_output.bg_spatial = SID_output.bg_spatial_pre_crop_rectify(crop_border_microlenses(1)*Nnum + 1 : end - crop_border_microlenses(2)*Nnum, crop_border_microlenses(3)*Nnum + 1 : end - crop_border_microlenses(4)*Nnum);
% end
%
% figure; imagesc(SID_output.bg_spatial); axis image; colorbar; title('Spatial background');
% print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_bg_spatial.png']), '-dpng', '-r300');
% figure; plot(SID_output.bg_temporal); title('Temporal background');
% print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_bg_temporal.png']), '-dpng', '-r300');

%% Compute standard-deviation image (std. image)
% disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Computing standard deviation image']);
% if Input.rectify
%     [SID_output.std_image, ~] = par_compute_std_image(Input.LFM_folder, Input.frames.step, SID_output.bg_temporal, SID_output.bg_spatial_pre_crop_rectify, ...
%         Input.frames.end, Input.x_offset, Input.y_offset, Input.dx, psf_ballistic.Nnum, Input.mask, Input.crop_border_microlenses);
% else
%     [SID_output.std_image, ~] = par_compute_std_image(Input.LFM_folder, Input.frames.step, SID_output.bg_temporal, SID_output.bg_spatial_pre_crop_rectify, ...
%         Input.frames.end, 0, 0, 0, 0, Input.mask, Input.crop_border_microlenses);
% end
%
% figure; imagesc(SID_output.std_image, [prctile(SID_output.std_image(:), 0) prctile(SID_output.std_image(:), 100.0)]); title('Stddev image'); axis image; axis ij; colorbar;
% print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_stddev_img.png']), '-dpng', '-r600');

%% load sensor movie
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Loading LFM movie']);
tic;
[sensor_movie,SID_output.movie_size] = read_sensor_movie(Input.LFM_folder, Input.x_offset, Input.y_offset, Input.dx, psf_ballistic.Nnum, Input.rectify, Input.frames, Input.mask, Input.crop_border_microlenses);
toc
%% Compute background and std-image

if Input.bg_sub
    [SID_output.bg_spatial,SID_output.bg_temporal]=rank_1_factorization(sensor_movie,Input.bg_iter);
else
    SID_output.bg_spatial = zeros(size(sensor_movie,1),1);
    SID_output.bg_temporal = zeros(1,size(sensor_movie,2));
end

SID_output.std_image=compute_std_image(sensor_movie,SID_output.bg_spatial,SID_output.bg_temporal);

SID_output.bg_spatial = reshape(SID_output.bg_spatial,SID_output.movie_size(1:2));
SID_output.std_image = reshape(SID_output.std_image,SID_output.movie_size(1:2));

figure; imagesc(SID_output.bg_spatial); axis image; colorbar; title('Spatial background');
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_bg_spatial.png']), '-dpng', '-r300');
figure; plot(SID_output.bg_temporal); title('Temporal background');
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_bg_temporal.png']), '-dpng', '-r300');

figure; imagesc(SID_output.std_image, [prctile(SID_output.std_image(:), 0) prctile(SID_output.std_image(:), 100.0)]); title('Stddev image'); axis image; axis ij; colorbar;
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_stddev_img.png']), '-dpng', '-r600');


%% Find cropping mask, leaving out areas with stddev as in background-only area

disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Finding crop space']);
if ~isfield(Input,'crop_params')
    disp('Find appropriate crop_params!')
    Input.crop_params = [0.2 0.6];
    flag1 = false;
    flag2 = false;
    flag = false;
else
    flag1 = true;
    flag2 = true;
    flag = true;
end

while max(~flag1,max(~flag2,flag))
    
    if Input.bg_sub
        img = SID_output.bg_spatial;
    else
        img = SID_output.std_image;
    end
    bg = img/max(img(:));
    Nnum = psf_ballistic.Nnum;
    SID_output.microlenses=img;
    for ix=1:size(SID_output.std_image,1)/Nnum
        for iy=1:size(SID_output.std_image,2)/Nnum
            SID_output.microlenses((ix-1)*Nnum+1:ix*Nnum,(iy-1)*Nnum+1:iy*Nnum)=SID_output.microlenses((ix-1)*Nnum+1:ix*Nnum,(iy-1)*Nnum+1:iy*Nnum)/norm(reshape(SID_output.microlenses((ix-1)*Nnum+1:ix*Nnum,(iy-1)*Nnum+1:iy*Nnum),1,[]));
        end
    end
    Inside = bg;
    h = fspecial('average', 3*psf_ballistic.Nnum);
    Inside=conv2(Inside,h,'same');
    Inside=max(Inside-quantile(Inside(:),Input.crop_params(1)),0);
    Inside=conv2(single(Inside>0),h,'same');
    SID_output.microlenses=Inside.*SID_output.microlenses;
    SID_output.microlenses=max(SID_output.microlenses-quantile(SID_output.microlenses(:),Input.crop_params(2)),0);
    if ~flag1
        figure(1);imagesc(Inside);
        drawnow expose
        flag1 = input('Does figure(1) give a good representation of the activity in the standard-deviation image? (yes=1,no=0)');
        if ~flag1
            disp(['The current value of Input.crop_params(1) is: ' num2str(Input.crop_params(1))]);
            Input.crop_params(1) = input('Enter new Value for Input.crop_params(1): ');
        end
    end
    if ~flag2
        figure(2);imagesc(SID_output.microlenses);
        drawnow expose
        flag2 = input('Does figure(2) give a good representation of the microlens pattern? (yes=1,no=0)');
        if ~flag2
            disp(['The current value of Input.crop_params(2) is: ' num2str(Input.crop_params(2))]);
            Input.crop_params(2) = input('Enter new Value for Input.crop_params(2): ');
        end
    end
    flag = false;
end

if Input.do_crop
    if ~isfield(Input,'crop_mask')
        Input.crop_mask=Inside;
    end
    [sensor_movie, SID_output] = crop(sensor_movie, SID_output,Inside,Input.crop_mask,Nnum);
else
    Inside = SID_output.std_image * 0 + 1;
    SID_output.idx=find(Inside>0);
end

timestr = datestr(now, 'YYmmddTHHMM');
figure;
hold on;
imagesc(Inside);
contour(Inside, [1e-10 1e-10], 'w');
axis ij;
colorbar();
axis image;
hold off;
print(fullfile(Input.output_folder, [timestr '_crop_mask.png']), '-dpng', '-r300');

%% de-trend
tic
if Input.detrend
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Detrending LFM movie']);
    SID_output.baseline_raw = squeeze(mean(sensor_movie,1))';
    %     delta=Input.delta;
    %     for t = 1 : size(SID_output.baseline_raw, 1)
    %         base(t) = min(SID_output.baseline_raw(max(1, t - delta) : min(size(SID_output.baseline_raw, 1), t + delta)));
    %     end
    %     base = double(base);
    if Input.delta <= 0
        smooth_window_span = numel(SID_output.baseline_raw) / max(1, abs(Input.delta));
    else
        smooth_window_span = 2 * Input.delta / Input.frames.step;
    end
    SID_output.baseline = smooth(SID_output.baseline_raw, smooth_window_span, 'sgolay', 3);
    figure; hold on; plot(SID_output.baseline_raw); plot(SID_output.baseline); title('Frame means (post bg subtract), raw + trend fit'); hold off;
    print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_trend_fit.pdf']), '-dpdf', '-r300');
    
    %fit_t_rng = Input.frames.start : Input.frames.step : ((size(sensor_movie,2)-1) * Input.frames.step + Input.frames.start);
    %SID_output.baseline_fit_params = exp2fit(fit_t_rng, base, 1);
    %SID_output.baseline_fit = SID_output.baseline_fit_params(1) + SID_output.baseline_fit_params(2) * exp(- (1 : num_frames) / SID_output.baseline_fit_params(3));
    %     figure; hold on; plot(SID_output.baseline); plot(SID_output.baseline_fit(Input.frames.start:Input.frames.step:Input.prime)); hold off; title('Smoothed frame means and trend fit');  %TODO: Input.prime doesn't seem to be the correct range here
    %     print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_trend_fit.pdf']), '-dpdf', '-r300');
    %
    sensor_movie = sensor_movie./SID_output.baseline';
    %TODO: check if trend fit worked, i.e. residuals are mostly gaussian
end
% sensor_movie_min = min(sensor_movie(:));
sensor_movie_max = max(sensor_movie(:));
% sensor_movie = (sensor_movie - sensor_movie_min) ./ (sensor_movie_max - sensor_movie_min);
sensor_movie = sensor_movie/sensor_movie_max;
toc

%% generate NNMF
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Generating rank-' num2str(Input.nnmf_opts.rank) '-factorization']);
p=0.8;
Input.nnmf_opts.ini_method='pca';
SID_output.neuron_centers_ini=[];
Input.nnmf_opts.active=SID_output.microlenses>0;
[SID_output.S, SID_output.T]=fast_NMF(max(sensor_movie-quantile(reshape(...
    sensor_movie(SID_output.microlenses==0,:),1,[]),p),0),Input.nnmf_opts);
SID_output.S=SID_output.S(:,~isoutlier(sum(SID_output.S,1),'ThresholdFactor',10));
if (~Input.optimize_kernel)&&(~isfield(Input.recon_opts,'ker_shape'))
    SID_output.S=[SID_output.S SID_output.std_image(:)]';
else
    SID_output.S=SID_output.S';
end

%% Crop sensor movie
sensor_movie = sensor_movie(SID_output.idx,:);

%% Plot NMF results
close all;
timestr = datestr(now, 'YYmmddTHHMM');
for i=1:size(SID_output.S, 1)
    figure( 'Position', [100 100 800 800]);%,'visible',false);
    subplot(4,1,[1,2,3]);
    imagesc(reshape(SID_output.S(i,:), size(SID_output.std_image)));
    axis image; colormap('parula'); colorbar;
    title(['NMF component ' num2str(i)]);
    subplot(4,1,4);
    plot(SID_output.T(i,:));
    print(fullfile(Input.output_folder, [timestr '_nnmf_component_' num2str(i, '%03d') '.png']), '-dpng', '-r600');
end
close all;

%% Save checkpoint
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Saving pre-nmf-recon checkpoint']);
save(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_checkpoint_pre-nmf-recon.mat']), 'Input', 'SID_output');

%% reconstruct spatial filters
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Reconstructing spatial filters']);
opts = Input.recon_opts;
opts.gpu_ids = Input.gpu_ids;
opts.microlenses = SID_output.microlenses;
SID_output.S = reshape(SID_output.S,[size(SID_output.S,1) SID_output.movie_size(1:2)]);

if Input.optimize_kernel
    if isfield(opts,'ker_param')
        opts=rmfield(opts,'ker_param');
    end
    kernel=0;
    while max(kernel(:))==0
        test_recon=reconstruct_S(SID_output.S(ceil(rand(1)*size(SID_output.S,1)),...
            :,:),psf_ballistic,opts);
        [kernel,SID_output.neur_rad]=find_kernel(test_recon{1},[1 1 4],...
            Input.neur_rad,Input.native_focal_plane,...
            Input.axial, Input.gpu_ids(1));
    end
    opts.ker_shape='user';
    opts.ker_param=kernel;
end

SID_output.recon = reconstruct_S(SID_output.S,psf_ballistic,opts);
SID_output.recon_opts = opts;

clear opts
%% crop reconstructed image with eroded mask, to reduce border artefacts
if numel(Input.mask) > 1 && any(Input.mask ~= 0)
    mask_dilated = imerode(Input.mask, strel('disk', 25));
    mask_dilated =  logical(ImageRect(double(mask_dilated), Input.x_offset, Input.y_offset, Input.dx, psf_ballistic.Nnum, ...
        true, Input.crop_border_microlenses(3), Input.crop_border_microlenses(4), Input.crop_border_microlenses(1), Input.crop_border_microlenses(2)));
    for i = 1:length(SID_output.recon)
        SID_output.recon{i} = SID_output.recon{i} .* mask_dilated;
    end
end

%% Plot reconstructed spatial filters
timestr = datestr(now, 'YYmmddTHHMM');
for i = 1:size(SID_output.S, 1)
    figure('Position', [50 50 1200 600]);
    subplot(1,4,[1:3])
    hold on;
    imagesc(squeeze(max(SID_output.recon{i}, [], 3)));
    axis image;
    axis ij;
    colorbar;
    hold off;
    subplot(1,4,4)
    imagesc(squeeze(max(SID_output.recon{i}, [], 2)));
    axis ij;
    colorbar;
    print(fullfile(Input.output_folder, [timestr '_nnmf_component_recon_' num2str(i, '%03d') '.png']), '-dpng', '-r600');
end
pause(10);
close all;

%% Save checkpoint
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Saving post-nmf-recon checkpoint']);
save(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_checkpoint_post-nmf-recon.mat']), 'Input', 'SID_output','-v7.3');

%% filter reconstructed spatial filters
opts.border = [1,1,15];
opts.gpu_ids = Input.gpu_ids;
if Input.optimize_kernel
    opts.neur_rad = 6;
else
    opts.neur_rad = Input.neur_rad;
end
opts.native_focal_plane = Input.native_focal_plane;

if Input.filter
    disp('Filtering reconstructed spatial filters');
    SID_output.segmm = filter_recon(SID_output.recon,opts);
else
    SID_output.segmm = SID_output.recon;
end

%% Segment reconstructed components
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': generate initial brain model'])

dim = [1 1 Input.axial];
SID_output.neuron_centers_ini = [];
[~,u] = max([size(SID_output.segmm,1),size(SID_output.segmm,2)]);
for ii=1:size(SID_output.segmm,u)
    SID_output.neuron_centers_per_component{ii} = segment_component(SID_output.segmm{ii},Input.segmentation.threshold);
    num(ii) = size(SID_output.neuron_centers_per_component{ii},1);
    disp(num(ii));
end

ids = isoutlier(num);
ids = (num>mean(num)).*ids;

for ii=find(ids)
    threshold = 0.1;
    SID_output.neuron_centers_per_component{ii} = segment_component(SID_output.segmm{ii},threshold);
    num(ii) = size(SID_output.neuron_centers_per_component{ii},1);
    disp(num(ii));
end

[SID_output.neuron_centers_ini,SID_output.neur_id]=iterate_cluster(SID_output.neuron_centers_per_component,Input.cluster_iter,Input.neur_rad,dim);

figure;plot(hist(SID_output.neuron_centers_ini(:,3),size(SID_output.recon{1},3)));
xlabel('z-axis');
ylabel('neuron frequency');
if ~isfield(Input.segmentation,'top_cutoff')
    disp('Check the axial distribution and remove top/bottom artefacts');
    Input.segmentation.top_cutoff = input('Input top cutoff \n');
end
if ~isfield(Input.segmentation,'bottom_cutoff')
    Input.segmentation.bottom_cutoff = input('Input bottom cutoff \n');
end
id=logical((SID_output.neuron_centers_ini(:,3)>=Input.segmentation.top_cutoff...
    ).*(SID_output.neuron_centers_ini(:,3)<=Input.segmentation.bottom_cutoff));
SID_output.neuron_centers_ini=SID_output.neuron_centers_ini(id,:);
SID_output.neur_id=SID_output.neur_id(id,:);

%% Plot segmentation result
timestr = datestr(now, 'YYmmddTHHMM');
for i = 1:numel(SID_output.segmm)
    figure('Position', [50 50 1200 600]);
    colormap parula;
    subplot(1,4,[1:3])
    hold on;
    imagesc(squeeze(max(SID_output.segmm{i}, [], 3)));
    scatter(SID_output.neuron_centers_per_component{i}(:,2), SID_output.neuron_centers_per_component{i}(:,1), 'r.');
    axis image;
    axis ij;
    colorbar;
    hold off;
    subplot(1,4,4)
    hold on;
    imagesc(squeeze(max(SID_output.segmm{i}, [], 2)));
    axis ij;
    scatter(SID_output.neuron_centers_per_component{i}(:,3), SID_output.neuron_centers_per_component{i}(:,1), 'r.');
    xlim([1 size(SID_output.segmm{i}, 3)]);
    ylim([1 size(SID_output.segmm{i}, 1)]);
    colorbar;
    print(fullfile(Input.output_folder, [timestr '_segmm_segmentation_' num2str(i, '%03d') '.png']), '-dpng', '-r300');
end

%%
clearvars -except sensor_movie Input SID_output mean_signal psf_ballistic Hsize m sensor_movie_max sensor_movie_min dim;

%% Initiate forward_model
%TODO: check performance of generate_forward_model() with matfile
%psf_ballistic = matfile(Input.psf_filename_ballistic);

if ~isfield(Input,'use_std_GLL')
    Input.use_std_GLL = false;
end

if isempty(Input.gpu_ids)||Input.use_std_GLL
    SID_output.forward_model_ini=generate_LFM_library_CPU(SID_output.neuron_centers_ini, psf_ballistic, round(SID_output.neur_rad), dim, size(SID_output.recon{1}));
else
    opts = SID_output.recon_opts;
    opts.NumWorkers=10;
    opts.image_size = SID_output.movie_size(1:2);
    opts.axial = Input.axial;
    opts.neur_rad = Input.neur_rad;
    SID_output.forward_model_ini=generate_LFM_library_GPU(SID_output.recon,SID_output.neuron_centers_ini,round(Input.neur_id),psf_ballistic,opts);
end

%% generate template
thres=0.01;
SID_output.template=generate_template(SID_output.neuron_centers_ini,psf_ballistic.H,SID_output.std_image,thres);

%% crop model
neur=find(squeeze(max(SID_output.forward_model_ini(:,SID_output.idx),[],2)>0));
SID_output.forward_model_iterated=SID_output.forward_model_ini(neur,SID_output.idx);
SID_output.neuron_centers_iterated=SID_output.neuron_centers_ini(neur,:);
SID_output.indices_in_orig=neur;

template_=SID_output.template(neur,SID_output.idx);
Nnum=psf_ballistic.Nnum;
% clearvars -except sensor_movie Input SID_output mean_signal template_ neur Nnum neur sensor_movie_max sensor_movie_min psf_ballistic;

%% SID-Alternative-convex-search
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Start optimizing model'])

tic
opts_temp=struct;
opts_spat=struct;
opts_temp.idx=SID_output.idx;
opts_temp.microlenses = SID_output.microlenses;
opts_spat.bg_sub = Input.bg_sub;
opts_temp.bg_sub = Input.bg_sub;
opts_temp.Nnum = Nnum;
% opts_spat.lambda=3e-3;

if isfield(Input, 'bg_sub') && Input.bg_sub
    SID_output.forward_model_iterated(end+1,:) = SID_output.bg_spatial(SID_output.idx);
    SID_output.indices_in_orig=[SID_output.indices_in_orig' length(SID_output.indices_in_orig)+1];
end

sensor_movie =double(sensor_movie*sensor_movie_max);
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Starting temporal update']);
SID_output.forward_model_iterated=(1./(sqrt(sum(SID_output.forward_model_iterated.^2....
    ,2)))).*SID_output.forward_model_iterated;
SID_output.timeseries_ini = LS_nnls(SID_output.forward_model_iterated(:,SID_output.microlenses(SID_output.idx)>0)', double(sensor_movie(SID_output.microlenses(SID_output.idx)>0,:)), opts_temp);
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Temporal update completed']);

SID_output.timeseries_iterated=SID_output.timeseries_ini;
toc

disp('---');
disp('---');

for iter=1:Input.num_iter
    disp([num2str(iter) '. iteration started']);
    
    [SID_output.timeseries_iterated,SID_output.forward_model_iterated,template_,...
        SID_output.indices_in_orig] = spatial_SID_update(...
        sensor_movie,SID_output.timeseries_iterated,...
        SID_output.forward_model_iterated,template_,...
        SID_output.indices_in_orig,opts_spat);
    
    if isfield(Input, 'update_template') && Input.update_template
        if iter>=2
            for neuron=1:size(template_,1)
                crop=zeros(size(SID_output.std_image));
                crop(SID_output.idx)=template_(neuron,:);
                img=reshape(crop,size(SID_output.std_image));
                img=conv2(img,ones(2*Nnum),'same')>0;
                img=img(:);
                template_(neuron,:)=(img(SID_output.idx)>0.1);
                disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' num2str(neuron)])
            end
        end
    end
    
    [SID_output.forward_model_iterated,SID_output.timeseries_iterated,template_...
        ,SID_output.indices_in_orig] = temporal_SID_update(...
        sensor_movie,SID_output.forward_model_iterated,SID_output.timeseries_iterated...
        ,template_,SID_output.indices_in_orig,opts_temp);
    

    [SID_output.forward_model_iterated,SID_output.timeseries_iterated,template_...
        ,SID_output.indices_in_orig] = merge_filters(...
        SID_output.forward_model_iterated,SID_output.timeseries_iterated...
        ,template_,SID_output.indices_in_orig,opts_temp);
    disp([num2str(iter) '. iteration completed']);
end
SID_output.neuron_centers_iterated=SID_output.neuron_centers_iterated(SID_output.indices_in_orig(1:end-1),:);

SID_output.template_iterated=template_;
opts_temp.warm_start=[];
clear sensor_movie;
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Model optimization completed']);

%%
forward_model=zeros(size(SID_output.forward_model_iterated,1),length(SID_output.std_image(:)));
forward_model(:,SID_output.idx)=SID_output.forward_model_iterated;
[SID_output.recon_NSF, x,y,z]=fast_NSF_recon(forward_model,ceil(SID_output.neuron_centers_iterated),psf_ballistic, size(SID_output.std_image));


%% extract time series at location LFM_folder
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Extracting Timeseries']);
opts_temp.warm_start=[];
% opts_temp.idx=SID_output.idx;
% opts_temp.max_iter=20000; % already defined in the last section
opts_temp.frame=Input.frames; %frames for model optimization;
opts_temp.outfile = fullfile(Input.output_folder, 'timeseries_debug_out.mat');
if isfield(Input, 'detrend') && Input.detrend
    opts_temp.baseline=SID_output.baseline;
end
tic
[timeseries_1, Varg] = incremental_temporal_update_gpu(SID_output.forward_model_iterated, Input.LFM_folder, [], Input.Junk_size, Input.x_offset,Input.y_offset,Input.dx,Nnum,opts_temp);
toc
SID_output.timeseries_total=zeros(size(timeseries_1,1),length(Varg));
SID_output.timeseries_total(:, Varg==1) = timeseries_1;
if nnz(Varg==0) > 0
    SID_output.timeseries_total(:, Varg==0) = SID_output.timeseries_iterated;
end
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Extraction complete']);

%% Signal2Noise ordering

%% save SID_output
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Saving result'])
SID_output.Input = Input;
save(fullfile(Input.output_folder, Input.SID_output_name), 'Input', 'SID_output', '-v7.3');

%% Summary figure: NNMF MIPs, with centers overlaid
timestr = datestr(now, 'YYmmddTHHMM');
nmf_mip = SID_output.recon{1};
for i=2:numel(SID_output.recon)
    nmf_mip = max(nmf_mip, SID_output.recon{i});
end

figure('Position', [50 50 1200 600]);
colormap parula;
subplot(1,4,[1:3])
hold on;
imagesc(squeeze(max(nmf_mip, [], 3)));
scatter(SID_output.neuron_centers_iterated(:,2), SID_output.neuron_centers_iterated(:,1), 'r.');
axis image;
title([Input.SID_output_name ' - NNMF components MIPs, with segmentation centers'], 'Interpreter', 'none');
colorbar;
hold off;
subplot(1,4,4)
hold on;
imagesc(squeeze(max(nmf_mip, [], 2)));
scatter(SID_output.neuron_centers_iterated(:,3), SID_output.neuron_centers_iterated(:,1), 'r.');
xlim([1 size(SID_output.recon{i}, 3)]);
ylim([1 size(SID_output.recon{i}, 1)]);
colorbar;
print(fullfile(Input.output_folder, [timestr '_nnmf_components_mip.png']), '-dpng', '-r300');

%% Timeseries, heatmap, clustered
timestr = datestr(now, 'YYmmddTHHMM');
figure('Position', [50 50 1200 600]);
ts = zscore(SID_output.timeseries_iterated, 0, 2);
clustered_ixs = clusterdata(ts, 'criterion', 'distance', 'distance', 'correlation', 'maxclust', floor(size(ts,1)/10));
tsi = [clustered_ixs ts];
ts = sortrows(tsi);
ts = ts(2:end,:);
limits = [prctile(ts(:), 0.01), prctile(ts(:), 99.9)];
imagesc(ts, limits);
title([Input.SID_output_name ' - timeseries, z-scored, corr-clustered'], 'Interpreter', 'none');
colormap parula;
colorbar;
print(fullfile(Input.output_folder, [timestr '_timeseries_zscore.png']), '-dpng', '-r300');

%% Timeseries, stacked (random subset of 100 traces)
ts = zscore(SID_output.timeseries_iterated, 0, 2);
y_shift = 4;
clip = true;
if size(ts,1) > 100
    sel = randperm(size(ts,1), 100);
else
    sel = 1:size(ts,1);
end
nixs = 1:size(ts,1);
sel_nixs = nixs(sel);

figure('Position', [10 10 2000 2000]);
title([Input.SID_output_name ' - timeseries, z-scored'], 'Interpreter', 'none');
subplot(121);
hold on
for n_ix = 1:floor(numel(sel_nixs)/2)
    ax = gca();
    ax.ColorOrderIndex = 1;
    loop_ts = ts(sel_nixs(n_ix),:);
    if clip
        loop_ts(loop_ts > 3*y_shift) = y_shift;
        loop_ts(loop_ts < -3*y_shift) = -y_shift;
    end
    t = (0:size(ts,2)-1);
    %plot(t, mat2gray(squeeze(loop_ts)) + 1*(n_ix-1));
    plot(t, squeeze(loop_ts) + y_shift*(n_ix-1));
    %text(30, y_shift*(n_ix-1), num2str(sel_sv(n_ix)));
end
xlabel('Frame');
%ylabel('Z-score');
%ylim([0 size(p.timeseries{1}, 1)]);
xlim([min(t) max(t)]);
%legend(p.labels, 'location', 'NorthEast');
hold off;
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
legend('boxoff');

subplot(122);
hold on;
for n_ix = ceil(numel(sel_nixs)/2):numel(sel_nixs)
    ax = gca();
    ax.ColorOrderIndex = 1;
    loop_ts = ts(sel_nixs(n_ix),:);
    if clip
        loop_ts(loop_ts > y_shift) = y_shift;
        loop_ts(loop_ts < -y_shift) = -y_shift;
    end
    t = (0:size(ts,2)-1);
    %plot(t, mat2gray(squeeze(loop_ts)) + 1*(n_ix-1));
    plot(t, squeeze(loop_ts) + y_shift*(n_ix-1));
    %text(30, y_shift*(n_ix-1), num2str(sel_sv(n_ix)));
end
xlabel('Frame');
%ylabel('Z-score');
%ylim([0 size(p.timeseries{1}, 1)]);
xlim([min(t) max(t)]);
%legend(p.labels, 'location', 'NorthEast');
hold off;
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
legend('boxoff');
print(fullfile(Input.output_folder, [timestr '_timeseries_zscore_stacked.png']), '-dpng', '-r300');

%% Inspect forward model and associated ts
ix = randperm(size(SID_output.timeseries_iterated, 1), 1);
fps = 16;
figure('Position', [20, 20, 2000, 2000]);
subplot(3,1,1:2);
imagesc(reshape(SID_output.forward_model_iterated(ix,:), size(SID_output.std_image)), [0 max(SID_output.forward_model_iterated(ix,:))]);
axis image;
colorbar();
subplot(3,1,3);
plot((1:size(SID_output.timeseries_iterated,2))/15, SID_output.timeseries_iterated(ix,:));

%% Delete cached psf file
if ~strcmp(Input.psf_cache_dir, '')
    disp([datestr(now,  'YYYY-mm-dd HH:MM:SS') ': Deleting cached psf file']);
    rmdir(Input.psf_cache_dir_unique, 's');
end

%%
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'main_nnmf_SID() returning'])
end
