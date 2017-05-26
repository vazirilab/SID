function main_nnmf_SID(indir, outdir, psffile, x_offset, y_offset, dx, optional_args)
%% fish timeseries extraction
% all Input fields used in the rest of the script:
% Input.LFM_folder
% Input.psf_filename_ballistic
% Input.output_folder
% Input.x_offset
% Input.y_offset
% Input.dx
% Input.rank
% Input.output_name
% Input.tmp_dir
% Input.step
% Input.bg_iter
% Input.rectify
% Input.Junk_size
% Input.bg_sub
% Input.prime
% Input.gpu_ids
% Input.num_iter
% Input.native_focal_plane
% Input.thres
% Input.nnmf_opts
% Input.recon_opts
% Input.update_template
% Input.detrend
% Input.fluoslide_fn
% Input.delta

% Input.frames.start = 1;%frames_for_model_optimization
% Input.frames.steps = 10;
% Input.frames.end = 1e6;

%% Required parameters
Input.LFM_folder = indir;
Input.psf_filename_ballistic = psffile;
Input.output_folder = outdir;
Input.x_offset = x_offset;
Input.y_offset = y_offset;
Input.dx = dx;

%% Optional parameters
if isfield(optional_args, 'rank')
    Input.rank = optional_args.rank;
else
    Input.rank = 30; % If Input.rank==0 SID classic instead of SID_nmf
end

if isfield(optional_args, 'out_filename')
    Input.output_name = optional_args.out_filename;
else
    Input.output_name = ['nnmf_sid_result_' datestr(now, 'YY-mm-ddTHHMM') '.mat'];
end

if isfield(optional_args, 'tmp_dir')
    Input.tmp_dir = optional_args.tmp_dir;
else
    Input.output_name = tempdir();
end

if isfield(optional_args, 'step')
    Input.step = optional_args.step;
else
    Input.step = 1;
end

if isfield(optional_args, 'step_')
    Input.step_ = optional_args.step_;
else
    Input.step_ = 3;
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

if isfield(optional_args, 'prime')
    Input.prime = optional_args.prime;
else
    Input.prime = inf;
end

if isfield(optional_args, 'frames')
    Input.frames = optional_args.frames;
else
   Input.frames.start = 1;
   Input.frames.steps = 10;
   Input.frames.end = inf;
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

if isfield(optional_args, 'delta')
    Input.delta = optional_args.delta;
else
    Input.delta = 200;
end

% typical neuron radius in px. Typically 6 for fish using 20x/0.5
% objective, 9-12 for mouse cortex and 16x/0.8
if isfield(optional_args, 'neuron_radius_px')
    Input.thres = optional_args.neuron_radius_px;
else
    Input.thres = 8;
end

if isfield(optional_args, 'recon_opts')
    Input.recon_opts = optional_args.recon_opts;
else
    Input.recon_opts.p=2;
    Input.recon_opts.maxIter=8;
    Input.recon_opts.mode='TV';
    Input.recon_opts.lambda=[ 0, 0, 10];
    Input.recon_opts.lambda_=0.1;
    Input.recon_opts.form='free';    
end
 
if isfield(Input,'filter')
    Input.filter = optional_args.filter;
else
    Input.filter=0;
end

if isfield(optional_args, 'frames_for_model_optimization')
    Input.frames_for_model_optimization = optional_args.frames_for_model_optimization;
else
    Input.frames_for_model_optimization.start = 1;
    Input.frames_for_model_optimization.step = 1;
    Input.frames_for_model_optimization.end = inf;
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

%%
do_crop = 1; %% Oliver suggest = 1
crop_thresh_coord_x = 0.8;	%values for fish
crop_thresh_coord_y = 0.75;	%values for fish
Input.nnmf_opts.max_iter = 300;
Input.nnmf_opts.lambda_t = 0;
Input.nnmf_opts.lambda_s = 0.1;
Input.update_template = false;
Input.detrend = false;
Input.de_trend = true;
Input.optimize_kernel = 0;

%%
psf_ballistic=matfile(Input.psf_filename_ballistic);
Input.fluoslide_fn = ['fluoslide_Nnum' num2str(psf_ballistic.Nnum) '.mat'];
if ~exist(Input.output_folder, 'dir')
    mkdir(Input.output_folder);
end

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
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Computing background components']);
if Input.bg_sub==1
    [output.bg_temporal,output.bg_spatial] = par_rank_1_factorization(Input.LFM_folder, Input.step, Input.bg_iter, 0, 0, 0, 0, Input.prime, Input.mask);
else
    output.bg_temporal=[];
    output.bg_spatial=[];
end
mkdir(Input.output_folder)
figure; imagesc(output.bg_spatial); axis image; colorbar; title('Spatial background');
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_bg_spatial.pdf']), '-dpdf', '-r300');
figure; plot(output.bg_temporal); title('Temporal background');
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_bg_temporal.pdf']), '-dpdf', '-r300');

%%% Compute mean_signal
%output.mean_signal=par_mean_signal(Input.LFM_folder,Input.step, Input.x_offset,Input.y_offset,Input.dx,psf_ballistic.Nnum,Input.prime);

%% Compute standard-deviation image (std. image)
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Computing standard deviation image']);
if Input.rectify==1
    [output.std_image,~]=par_compute_std_image(Input.LFM_folder,Input.step,output.bg_temporal,output.bg_spatial,Input.prime, Input.x_offset,Input.y_offset,Input.dx,psf_ballistic.Nnum, Input.mask);
else
    [output.std_image,~]=par_compute_std_image(Input.LFM_folder,Input.step,output.bg_temporal,output.bg_spatial,Input.prime, 0, 0, 0, 0, Input.mask);
end

if (Input.bg_sub==1)&&(Input.rectify==1)
    output.bg_spatial = ImageRect(output.bg_spatial, Input.x_offset, Input.y_offset, Input.dx, psf_ballistic.Nnum, 0);
end


figure; imagesc(output.std_image, [prctile(output.std_image(:), 0) prctile(output.std_image(:), 100.0)]); axis image; colorbar;
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_stddev_img.png']), '-dpng', '-r300');

%% load sensor movie
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Loading LFM movie']);
tic;
[sensor_movie,num_frames]=read_sensor_movie(Input.LFM_folder,Input.x_offset,Input.y_offset,Input.dx,psf_ballistic.Nnum,Input.rectify,Input.frames_for_model_optimization, Input.mask);
if isinf(Input.prime)
    Input.prime=num_frames;
end
toc

%% find crop space
if do_crop
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Finding crop space']);
    Inside=output.std_image(ceil(crop_thresh_coord_x * size(output.std_image,1)):end, ceil(crop_thresh_coord_y * size(output.std_image,2)):end);
    Inside=output.std_image-mean(Inside(:))-2*std(Inside(:));
    Inside(Inside<0)=0;
    beads=bwconncomp(Inside>0);
    for kk=1:beads.NumObjects
        if numel(beads.PixelIdxList{kk})<8
            Inside(beads.PixelIdxList{kk})=0;
        end
    end
    h = fspecial('average', 3*psf_ballistic.Nnum);
    Inside=conv2(Inside,h,'same');
    beads=bwconncomp(Inside);                    %reduce to biggest connected component
    for kk=2:beads.NumObjects
        Inside(beads.PixelIdxList{kk})=0;
    end
    output.idx=find(Inside>0);
else
    Inside = output.std_image * 0 + 1;
end
output.idx=find(Inside>0);

timestr = datestr(now, 'YYmmddTHHMM');
figure;
imagesc(Inside);
colorbar();
axis image;
print(fullfile(Input.output_folder, [timestr '_crop_mask' num2str(i, '%03d') '.png']), '-dpng', '-r300');

%% subtract baseline outside of brain
outside = ~Inside;
if do_crop
    baseline = squeeze(mean(sensor_movie(logical(outside),:),1));
    for ix=1:size(sensor_movie,2)
        sensor_movie(:,ix)= sensor_movie(:,ix) - baseline(ix);
    end
    sensor_movie(sensor_movie<0)=0;
end

%% de-trend
tic
if Input.de_trend
    disp('Detrending LFM movie');
    output.baseline=squeeze(mean(sensor_movie,1));
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Detrending LFM movie']);
    figure; plot(output.baseline); title('Frame means after background subtraction');
    output.baseline=smooth(output.baseline,70);
    delta=Input.delta;
    for t=1:size(output.baseline,1)
        base(t) = min(output.baseline(max(1,t-delta):min(size(output.baseline,1),t+delta)));
    end
    base = double(base);
    fit_t_rng = Input.frames_for_model_optimization.start : Input.frames_for_model_optimization.step : (size(sensor_movie,2)-1)*Input.frames_for_model_optimization.step + Input.frames_for_model_optimization.start;
    output.baseline_fit_params = exp2fit(fit_t_rng, base, 1);
    output.baseline_fit = output.baseline_fit_params(1) + output.baseline_fit_params(2) * exp(- (1 : Input.prime) / output.baseline_fit_params(3));
    figure; hold on; plot(output.baseline); plot(output.baseline_fit); hold off; title('Smoothed frame means and trend fit');  %TODO: Input.prime doesn't seem to be the correct range here
    print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_trend_fit.pdf']), '-dpdf', '-r300');
    sensor_movie = sensor_movie * diag(1 ./ output.baseline_fit);
end
sensor_movie_min = min(sensor_movie(:));
sensor_movie_max = max(sensor_movie(:));
sensor_movie = (sensor_movie - sensor_movie_min) ./ (sensor_movie_max - sensor_movie_min);
toc

%% generate NNMF
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Generating rank-' num2str(Input.rank) '-factorization']);
ops.bg_temporal=squeeze(mean(sensor_movie,1));
Input.nnmf_opts.bg_temporal=squeeze(mean(sensor_movie,1));
output.centers=[];
[S, T]=fast_NMF_2(sensor_movie,Input.rank,Input.nnmf_opts);
S=[S output.std_image(:)]';
sensor_movie = sensor_movie(output.idx,:);
output.S = S;
output.T = T;

%% Plot NMF results
close all;
timestr = datestr(now, 'YYmmddTHHMM');
for i=1:size(output.T, 1)
    figure( 'Position', [100 100 800 800]);
    subplot(4,1,[1,2,3]);
    imagesc(reshape(output.S(i,:), size(Inside))); axis image; colormap('parula'); colorbar;
    title(['NMF component ' num2str(i)]);
    subplot(4,1,4);
    plot(output.T(i,:));
    print(fullfile(Input.output_folder, [timestr '_nnmf_component_' num2str(i, '%03d') '.png']), '-dpng', '-r300');
    close;
end

%%
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

%% reconstruct spatial filters
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Reconstructing spatial filters']);
poolobj = gcp('nocreate');
delete(poolobj);

if isempty(Input.gpu_ids)
    infile=struct;
    for k=1:size(S,1)
        img_=reshape(S(k,:),size(output.std_image,1),[]);
        img_=img_/max(img_(:));
        img_=img_-mean(mean(img_(ceil(0.8*size(output.std_image,1)):end,ceil(0.75*size(output.std_image,2)):end))); % TN TODO: hardcoded vals
        img_(img_<0)=0;
        infile.LFmovie=full(img_)/max(img_(:));
        output.recon{k} = reconstruction_cpu_sparse(Input.psf_filename_ballistic, infile, Input.recon_opts);
        disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' k]);
    end   
else
    nn=length(Input.gpu_ids);
    gimp=Input.gpu_ids;
    parpool(nn);
    
    if Input.optimize_kernel
        infile=struct;
        options{1}=Input.recon_opts;
        options{1}.gpu_ids=gimp(1);
        
        img_=reshape(S(1,:),size(output.std_image,1),[]);
        img_=img_/max(img_(:));
        img_=img_-mean(mean(img_(ceil(0.8*size(output.std_image,1)):end,ceil(0.75*size(output.std_image,2)):end)));
        img_(img_<0)=0;
        infile.LFmovie=full(img_)/max(img_(:));        
        test = reconstruction_new(infile, Input.psf_filename_ballistic, options{1});
        [~,kernel] = total_deconv(test,Input.total_deonv_opts);
        Input.form = 'free';
        Input.recon_opts.rad=kernel;
    end
    
    for kk=1:nn:size(S,1)
        img=cell(nn,1);
        for worker=1:min(nn,size(S,1)-(kk-1))
            k=kk+worker-1;
            img_=reshape(S(k,:),size(output.std_image,1),[]);
            img_=img_/max(img_(:));
            img_=img_-mean(mean(img_(ceil(0.8*size(output.std_image,1)):end,ceil(0.75*size(output.std_image,2)):end))); % TN TODO: hardcoded vals
            
            img_(img_<0)=0;
            img{worker}=full(img_)/max(img_(:));
        end
        options=cell(min(nn,size(S,1)-(kk-1)),1);
        recon=cell(min(nn,size(S,1)-(kk-1)),1);
        tmp_recon_opts = Input.recon_opts;
        parfor worker=1:min(nn,size(S,1)-(kk-1))
            infile=struct;
            infile.LFmovie=(img{worker});
            options{worker}=tmp_recon_opts;
            options{worker}.gpu_ids=mod((worker-1),nn)+1;
            options{worker}.gpu_ids=gimp(options{worker}.gpu_ids);
            recon{worker}= reconstruction_sparse(infile, Input.psf_filename_ballistic, options{worker});
            gpuDevice([]);
        end
        for kp=1:min(nn,size(S,1)-(kk-1))
            output.recon{kk+kp-1}=recon{kp};
        end
        disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' num2str(kk)])
    end
end

%% Plot reconstructed spatial filters
timestr = datestr(now, 'YYmmddTHHMM');
for i = 1:length(output.recon)
    figure('Position', [50 50 1200 600]); 
    subplot(1,4,[1:3])
    hold on;
    imagesc(squeeze(max(output.recon{i}, [], 3)));
    axis image;
    colorbar;
    hold off;
    subplot(1,4,4)
    imagesc(squeeze(max(output.recon{i}, [], 2)));
    colorbar;
    print(fullfile(Input.output_folder, [timestr '_nnmf_component_recon_' num2str(i, '%03d') '.png']), '-dpng', '-r300');
end
%pause(10);
%close all;

%% Save checkpoint
save(fullfile(Input.output_folder, 'checkpoint_post-nmf-recon.mat'), 'Input', 'output');

%% filter reconstructed spatial filters
if Input.filter
    disp('Filtering reconstructed spatial filters');
    Hsize = size(psf_ballistic.H);
    m=[size(output.std_image,1),size(output.std_image,2),Hsize(5)];
    bordz = 15;
    bord=1;
    cellSize = 14;
    gpu = ~isempty(Input.gpu_ids);
    [X,Y,Z]=meshgrid(1:2:2*size(output.std_image,2)-1,1:2:2*size(output.std_image,1)-1,[1:Input.native_focal_plane-1 Input.native_focal_plane+1:Hsize(5)]);
    [Xq,Yq,Zq]=meshgrid(1:2:2*size(output.std_image,2)-1,1:2:2*size(output.std_image,1)-1,[1:Hsize(5)]);
    
    for kk=1:nn:size(S,1)
        img=cell(nn,1);
        for worker=1:min(nn,size(S,1)-(kk-1))
            k=kk+worker-1;
            V=interp3(X,Y,Z,output.recon{k}(:,:,[1:Input.native_focal_plane-1 Input.native_focal_plane+1:Hsize(5)]),Xq,Yq,Zq);
            
            I=zeros(size(V)+[0 0 2*bordz],'single');
            I(:,:,bordz+1:bordz+Hsize(5))=single(V);
            for k=0:bordz-1
                I(:,:,bordz-k)=I(:,:,bordz+1-k)*0.96;
                I(:,:,bordz+Hsize(5)+k)=I(:,:,bordz+Hsize(5)-1+k)*0.96;
            end
            Ifiltered = I/max(I(:));
            img{worker}=full(Ifiltered);
        end
        segm_=zeros(min(nn,size(S,1)-(kk-1)),size(Ifiltered,1)-2*bord+1,size(Ifiltered,2)-2*bord+1,Hsize(5));
        parfor worker=1:min(nn,size(S,1)-(kk-1))
            filtered_Image_=band_pass_filter(img{worker}, cellSize, 8, gimp(worker),1.2);
            segm_(worker,:,:,:)=filtered_Image_(bord:size(filtered_Image_,1)-bord,bord:size(filtered_Image_,2)-bord,bordz+1:bordz+Hsize(5));                      
            if gpu
                gpuDevice([]);
            end
        end
        for kp=1:min(nn,size(S,1)-(kk-1))
            filtered_Image=zeros(size(Ifiltered)-[0 0 2*bordz]);
            filtered_Image(bord:size(Ifiltered,1)-bord,bord:size(Ifiltered,2)-bord,:)=squeeze(segm_(kp,:,:,:));
            output.segmm{kk+kp-1}=filtered_Image;
        end
        disp(kk)
    end
    
    for ix=1:size(S,1)
        Vol = output.segmm{1}*0;
        Vol(bordz:end-bordz,bordz:end-bordz,:) = output.segmm{ix}(bordz:end-bordz,bordz:end-bordz,:);
        output.segmm{ix} = Vol;
    end
else
    output.segmm = output.recon;
end

%% Segment reconstructed components
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': generate initial brain model'])
output.centers=[];
for ii=1:size(output.segmm,2)
    segm=output.segmm{ii};
    for kk=1:size(segm,3)
        segm(:,:,kk)=segm(:,:,kk).*(Inside>0);
    end
    segm=segm/max(segm(:));
    segm=segm-0.02; % Tobias, this should be made in to a parameter so that it can be adjust flexiably,
    segm(segm<0)=0;
    centers=[];
    B=reshape(segm,[],1);
    beads=bwconncomp((segm));
    for k=1:beads.NumObjects
        qu=B(beads.PixelIdxList{1,k});
        q=sum(B(beads.PixelIdxList{1,k}));
        [a,b,c]=ind2sub(size(segm),beads.PixelIdxList{1,k});
        centers(k,:)=([a,b,c]'*qu/q)';
    end
    output.centers_per_component{ii} = centers;
    size(centers)
    if (ii==1)
        output.centers=centers;
    else
        id=[];
        for k=1:size(centers,1)
            flag=1;
            for j=1:size(output.centers,1)
                if norm((output.centers(j,:)-centers(k,:))*diag([1 1 4]))<Input.thres
                    flag=0;
                end
            end
            if flag
                id=[id k];
            end
        end
        output.centers=[output.centers' centers(id,:)']';
    end
    
    
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Segmentation of component ' num2str(ii) ' resulted in ' num2str(size(output.centers, 1)) ' neuron candidates']);
end

segm=0*output.recon{1};
for ii=1:size(output.centers,1)
    segm(ceil(output.centers(ii,1)),ceil(output.centers(ii,2)),ceil(output.centers(ii,3)))=1;
end

%% Plot segmentation result
timestr = datestr(now, 'YYmmddTHHMM');
for i = 1:numel(output.segmm)
    figure('Position', [50 50 1200 600]); 
    colormap parula;
    subplot(1,4,[1:3])
    hold on;
    imagesc(squeeze(max(output.segmm{i}, [], 3)));
    scatter(output.centers_per_component{i}(:,2), output.centers_per_component{i}(:,1), 'r.');
    axis image;
    colorbar;
    hold off;
    subplot(1,4,4)
    hold on;
    imagesc(squeeze(max(output.segmm{i}, [], 2)));
    scatter(output.centers_per_component{i}(:,3), output.centers_per_component{i}(:,1), 'r.');
    xlim([1 size(output.segmm{i}, 3)]);
    ylim([1 size(output.segmm{i}, 1)]);
    colorbar;
    print(fullfile(Input.output_folder, [timestr '_segmm_segmentation_' num2str(i, '%03d') '.png']), '-dpng', '-r300');
end

%%
clearvars -except sensor_movie Input output mean_signal psf_ballistic Hsize m

%% Initiate forward_model
psf_ballistic=load(Input.psf_filename_ballistic);

output.forward_model=generate_foward_model(output.centers,psf_ballistic,8,3,size(output.recon{1})); %replace 8 by 1 if _7r psf

%% generate template
output.template=generate_template(output.forward_model,psf_ballistic.Nnum,0.005,size(output.std_image));

%% crop model
neur=find(squeeze(max(output.forward_model(:,output.idx),[],2)>0));
output.forward_model_=output.forward_model(neur,output.idx);

template_=output.template(neur,output.idx);
Nnum=psf_ballistic.Nnum;
clearvars -except sensor_movie Input output mean_signal template_ neur Nnum neur

%% optimize model
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Start optimizing model'])

tic
opts=[];
opts.tol=1e-2; 
opts.tol_=5*1e-1;
opts.gpu_ids=1;
opts.sample=600;
opts.display='off';
opts.gpu='off';
optz.solver=1;
optz.display='off';
optz.bg_sub=Input.bg_sub;
opts.max_iter=10000;
opts.idx=output.idx;
opts.lambda = 0;

if isfield(Input, 'bg_sub') && Input.bg_sub
%     bg_spatial_=average_ML(reshape(output.bg_spatial,size(output.bg_spatial)),Nnum, Input.fluoslide_fn);
%     bg_spatial_=bg_spatial_(output.idx);
%     bg_spatial_=bg_spatial_/norm(bg_spatial_(:));
    output.forward_model_(end+1,:) = output.bg_spatial(output.idx);
end

sensor_movie = double(sensor_movie);
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Starting temporal update'])
output.timeseries = fast_nnls(output.forward_model_', double(sensor_movie), opts);
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Temporal update completed']);

output.timeseries_=output.timeseries;
output.centers_=output.centers;
toc

disp('---');
disp('---');


%%
for iter=1:Input.num_iter
    id2=[];
    disp([num2str(iter) '. iteration started']);
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Pruning neurons']);
    for k=1:size(output.forward_model_,1)
        trace=output.timeseries_(k,:)>1e-7;
        if sum(trace)>1
            id2=[id2 k];
        end
    end
    
   
    output.timeseries_=output.timeseries_(id2,:);
    template_=template_(id2(1:end-Input.bg_sub),:);
    output.centers_=output.centers_(id2(1:end-Input.bg_sub),:);
    output.forward_model=output.forward_model(id2(1:end-Input.bg_sub),:);
    tic
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Starting spatial update']);
    output.timeseries_=diag(1./(sqrt(sum(output.timeseries_.^2,2))))*output.timeseries_;
    output.forward_model_=update_spatial_component(output.timeseries_, sensor_movie, template_, optz);
    toc
   
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Spatial update completed'])
    
    if isfield(Input, 'update_template') && Input.update_template
        if iter==2
            for neuron=1:size(template_,1)
                crop=zeros(size(output.std_image));
                crop(output.idx)=template_(neuron,:);
                img=reshape(crop,size(output.std_image));
                img=conv2(img,ones(2*Nnum),'same')>0;
                img=img(:);
                template_(neuron,:)=(img(output.idx)>0.1);
                disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' num2str(neuron)])
            end
        end
    end      
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Pruning neurons']);  
    id2=[];
    for k=1:size(output.forward_model_,1)
        trace=output.forward_model_(k,:)>1e-12;
        if sum(trace)>(Nnum^2)/3
            id2=[id2 k];
        end
        %         disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' k]);
    end
    output.timeseries_=output.timeseries_(id2,:);
    output.forward_model_=output.forward_model_(id2,:);
    template_=template_(id2(1:end-Input.bg_sub),:);
    output.centers=output.centers(id2(1:end-Input.bg_sub),:);
    output.forward_model=output.forward_model(id2(1:end-Input.bg_sub),:);
    tic
%     output.forward_model_=diag(1./(sqrt(sum(output.forward_model_.^2,2))))*output.forward_model_;
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Starting Temporal update']);
    opts.warm_start=output.timeseries_;
    output.timeseries_=fast_nnls(output.forward_model_',sensor_movie,opts);
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Temporal update completed']);
    toc
%     if mod(iter, 50) == 0
%         disp([num2str(iter) '. iteration completed']);
%     end
    disp([num2str(iter) '. iteration completed']);
end
output.template_=template_;
opts.warm_start=[];
clear sensor_movie;
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Model optimization completed']);

%% extract time series at location LFM_folder
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Extracting Timeseries']);
opts.step=Input.step;
opts.prime=Input.prime;
opts.warm_start=[];
% opts.idx=output.idx;
% opts.max_iter=20000; % already defined in the last section
opts.frame=Input.frames; %frames for model optimization;
opts.outfile = fullfile(Input.output_folder, 'timeseries_debug_out.mat');
if isfield(Input, 'detrend') && Input.detrend
    opts.baseline=output.baseline;
end
tic
[timeseries_1, Varg] = incremental_temporal_update_gpu(output.forward_model_, Input.LFM_folder, [], Input.Junk_size, Input.x_offset,Input.y_offset,Input.dx,Nnum,opts);
toc
output.timeseries_total=zeros(size(timeseries_1,1),length(Varg));
output.timeseries_total(:, Varg==1) = timeseries_1;
if nnz(Varg==0) > 0
    output.timeseries_total(:, Varg==0) = output.timeseries_;
end
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Extraction complete']);

%% save output
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Saving result'])
output.Input = Input;
save(fullfile(Input.output_folder, Input.output_name), 'Input', 'output', '-v7.3');

%% Summary figure: NNMF MIPs, with centers overlaid
timestr = datestr(now, 'YYmmddTHHMM');
nmf_mip = output.recon{1};
for i=2:numel(output.recon)
    nmf_mip = max(nmf_mip, output.recon{i});
end

figure('Position', [50 50 1200 600]);
colormap parula;
subplot(1,4,[1:3])
hold on;
imagesc(squeeze(max(nmf_mip, [], 3)));
scatter(output.centers_(:,2), output.centers_(:,1), 'r.');
axis image;
title([Input.output_name ' - NNMF components MIPs, with segmentation centers'], 'Interpreter', 'none');
colorbar;
hold off;
subplot(1,4,4)
hold on;
imagesc(squeeze(max(nmf_mip, [], 2)));
scatter(output.centers_(:,3), output.centers_(:,1), 'r.');
xlim([1 size(output.recon{i}, 3)]);
ylim([1 size(output.recon{i}, 1)]);
colorbar;
print(fullfile(Input.output_folder, [timestr '_nnmf_components_mip.png']), '-dpng', '-r300');

%% Timeseries, heatmap, clustered
timestr = datestr(now, 'YYmmddTHHMM');
figure('Position', [50 50 1200 600]);
ts = zscore(output.timeseries_, 0, 2);
clustered_ixs = clusterdata(ts, 'criterion', 'distance', 'distance', 'correlation', 'maxclust', floor(size(ts,1)/10));
tsi = [clustered_ixs ts];
ts = sortrows(tsi);
ts = ts(2:end,:);
limits = [prctile(ts(:), 0.01), prctile(ts(:), 99.9)];
imagesc(ts, limits);
title([Input.output_name ' - timeseries, z-scored, corr-clustered'], 'Interpreter', 'none');
colormap parula;
colorbar;
print(fullfile(Input.output_folder, [timestr '_timeseries_zscore.png']), '-dpng', '-r300');

%% Timeseries, stacked (random subset of 100 traces)
ts = zscore(output.timeseries_, 0, 2);
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
title([Input.output_name ' - timeseries, z-scored'], 'Interpreter', 'none');
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
ix = randperm(size(output.timeseries_,1), 1);
fps = 15;
figure('Position', [20, 20, 2000, 2000]); 
subplot(3,1,1:2);
imagesc(reshape(output.forward_model_(ix,:), size(output.std_image)), [0 max(output.forward_model_(ix,:))]); 
axis image;
colorbar();
subplot(3,1,3);
plot((1:size(output.timeseries_,2))/15, output.timeseries_(ix,:));

%% Delete cached psf file
if ~strcmp(Input.psf_cache_dir, '')
    disp([datestr(now,  'YYYY-mm-dd HH:MM:SS') ': Deleting cached psf file']);
    rmdir(Input.psf_cache_dir_unique, 's');
end

%%
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'main_nnmf_SID() returning'])
end
