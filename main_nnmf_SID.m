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
% Input.step_
% Input.bg_iter
% Input.rectify
% Input.Junk_size
% Input.bg_sub
% Input.prime
% Input.prime_
% Input.gpu_ids
% Input.num_iter
% Input.native_focal_plane
% Input.thres
% Input.nnmf_opts
% Input.recon_opts
% Input.update_template
% Input.detrend
% Input.fluoslide_fn

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
    Input.prime = 40000;
end

if isfield(optional_args, 'prime_')
    Input.prime_ = optional_args.prime_;
else
    Input.prime_ = 4800;
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
    Input.thres = optional_args.neuron_radius_px;
else
    Input.thres = 10;
end

if isfield(optional_args, 'recon_opts')
    Input.recon_opts = optional_args.recon_opts;
else
    Input.recon_opts.p = 2;
    Input.recon_opts.maxIter = 8;
    Input.recon_opts.mode = 'TV';
    Input.recon_opts.lambda = [0, 0, 10];
    Input.recon_opts.lambda_ = 0.1;
end

%%
do_crop = 0;
crop_thresh_coord_x = 0.5;
crop_thresh_coord_y = 0.5;
Input.nnmf_opts.max_iter = 1000;
Input.nnmf_opts.lambda = 0.1;
Input.update_template = false;
Input.detrend = false;

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

%% Compute bg components via rank-1-factorization
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Computing background components']);
if Input.bg_sub==1
    [output.bg_temporal,output.bg_spatial]=par_rank_1_factorization(Input.LFM_folder,Input.step, Input.bg_iter,Input.prime);
else
    output.bg_temporal=[];
    output.bg_spatial=[];
end
figure; imagesc(output.bg_spatial); axis image; colorbar; title('Spatial background');
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_bg_spatial.pdf']), '-dpdf', '-r300');
figure; plot(output.bg_temporal); title('Temporal background');
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_bg_temporal.pdf']), '-dpdf', '-r300');

%% Compute standard-deviation image (std. image)
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Computing standard deviation image']);
if Input.rectify==1
    [output.std_image,~]=par_compute_std_image(Input.LFM_folder,Input.step,output.bg_temporal,output.bg_spatial,Input.prime, Input.x_offset,Input.y_offset,Input.dx,psf_ballistic.Nnum);
else
    [output.std_image,~]=par_compute_std_image(Input.LFM_folder,Input.step,output.bg_temporal,output.bg_spatial,Input.prime);
end

if (Input.bg_sub==1)&&(Input.rectify==1)
    output.bg_spatial =  ImageRect(output.bg_spatial, Input.x_offset, Input.y_offset, Input.dx, psf_ballistic.Nnum,0);
end


figure; imagesc(output.std_image, [prctile(output.std_image(:), 0) prctile(output.std_image(:), 99.5)]); axis image; colorbar;
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_stddev_img.png']), '-dpng', '-r300');

%% load sensor movie and de-trend
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Loading LFM movie']);
tic;
sensor_movie=read_sensor_movie(Input.LFM_folder,Input.x_offset,Input.y_offset,Input.dx,psf_ballistic.Nnum,Input.rectify,Input.frames);
toc

tic;
baseline=mean(sensor_movie,1)';
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Detrending LFM movie']);
figure; plot(baseline); title('Frame means after background subtraction');

[baseline_fit, gof, ~] = fit((1:length(baseline))', baseline, 'poly3');
disp(baseline_fit);
disp(gof);
if gof.adjrsquare < 0.8
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'WARNING: Goodness of baseline fit seems bad. De-trending disabled.']);
    baseline_fit_vals = ones(size(sensor_movie,2), 1) * mean(baseline(:));
else
    baseline_fit_vals = feval(baseline_fit, 1:size(sensor_movie,2));
end
figure; hold on; plot(baseline); plot(baseline_fit_vals); hold off; title('Frame means and trend fit');
print(fullfile(Input.output_folder, [datestr(now, 'YYmmddTHHMM') '_trend_fit.pdf']), '-dpdf', '-r300');
sensor_movie = sensor_movie * diag(1./baseline_fit_vals);
sensor_movie_min = min(sensor_movie(:));
sensor_movie_max = max(sensor_movie(:));
sensor_movie = (sensor_movie - sensor_movie_min) ./ (sensor_movie_max - sensor_movie_min);
toc

%% find crop space
if do_crop
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Finding crop space']);
    sub_image=output.std_image(ceil(crop_thresh_coord_x * size(output.std_image,1)):end, ceil(crop_thresh_coord_y * size(output.std_image,2)):end);
    sub_image=output.std_image-mean(sub_image(:))-2*std(sub_image(:));
    sub_image(sub_image<0)=0;
    beads=bwconncomp(sub_image>0);
    for kk=1:beads.NumObjects
        if numel(beads.PixelIdxList{kk})<8
            sub_image(beads.PixelIdxList{kk})=0;
        end
    end
    h = fspecial('average', 2*psf_ballistic.Nnum);
    sub_image=conv2(sub_image,h,'same');
else
    sub_image = output.std_image * 0 + 1;
end
output.idx=find(sub_image>0);

%% generate NNMF
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Generating rank-' num2str(Input.rank) '-factorization']);
output.centers=[];
Input.nnmf_opts.bg_temporal=squeeze(mean(sensor_movie,1));
[S, T] = fast_NMF(sensor_movie, Input.rank, Input.nnmf_opts);
S=[S' output.std_image(:)]';
sensor_movie=sensor_movie(output.idx,:);
output.S = S;
output.T = T;

%% Plot NMF results
timestr = datestr(now, 'YYmmddTHHMM');
for i=1:size(output.T, 1)
    figure( 'Position', [100 100 800 800]);
    subplot(4,1,[1,2,3]);
    imagesc(reshape(output.S(i,:), size(sub_image))); axis image; colormap('parula'); colorbar;
    title(['NMF component ' num2str(i)]);
    subplot(4,1,4);
    plot(output.T(i,:));
    print(fullfile(Input.output_folder, [timestr '_nnmf_component_' num2str(i, '%03d') '.png']), '-dpng', '-r300');
end

%% reconstruct spatial filters
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Reconstructing spatial filters']);
psf_ballistic=load(Input.psf_filename_ballistic);
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
        output.recon{k} = reconstruction_cpu_sparse(psf_ballistic,infile, Input.recon_opts);
        disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' k]);
    end   
else
    nn=length(Input.gpu_ids);
    gimp=Input.gpu_ids;
    parpool(nn);
    
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
        
        recon=cell(min(nn,size(S,1)-(kk-1)),1);
        tmp_recon_opts = Input.recon_opts;
        parfor worker=1:min(nn,size(S,1)-(kk-1))
            infile=struct;
            infile.LFmovie=(img{worker});
            options=cell(min(nn,size(S,1)-(kk-1)),1);
            options{worker}=tmp_recon_opts;
            options{worker}.gpu_ids=mod((worker-1),nn)+1;
            options{worker}.gpu_ids=gimp(options{worker}.gpu_ids); %#ok<PFBNS>
            
            recon{worker}= reconstruction_sparse(infile, psf_ballistic, options{worker});
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
    imagesc(squeeze(max(output.recon{i}, [], 3)));
    axis image;
    colorbar;
    subplot(1,4,4)
    imagesc(squeeze(max(output.recon{i}, [], 2)));
    colorbar;
    print(fullfile(Input.output_folder, [timestr '_nnmf_component_recon_' num2str(i, '%03d') '.png']), '-dpng', '-r300');
end

%% generate initial brain model
output.centers=[];
for ii=1:size(output.recon,2)
    segm=output.recon{ii};
    for kk=1:size(segm,3)
       segm(:,:,kk)=segm(:,:,kk).*(sub_image>0); 
    end
    segm=segm/max(segm(:));
    segm=segm-0.01;
    segm(segm<0)=0;
    centers=[];
    B=reshape(segm,[],1);
    beads=bwconncomp(imregionalmax(segm));
    for k=1:beads.NumObjects
        qu=B(beads.PixelIdxList{1,k});
        q=sum(B(beads.PixelIdxList{1,k}));
        [a,b,c]=ind2sub(size(segm),beads.PixelIdxList{1,k});
        centers(k,:)=([a,b,c]'*qu/q)';
    end
    
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
    
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' num2str(ii)]);
end

segm=0*output.recon{1};
for ii=1:size(output.centers,1)
    segm(ceil(output.centers(ii,1)),ceil(output.centers(ii,2)),ceil(output.centers(ii,3)))=1;
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

disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Starting temporal update'])
output.timeseries = fast_nnls(output.forward_model_', sensor_movie, opts);
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Temporal update completed']);

output.timeseries_=output.timeseries;
output.centers_=output.centers;
toc

disp('---');
disp('---');



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
%TODO: define opts.frame here?
opts.outfile = fullfile(Input.output_folder, 'timeseries_debug_out.mat');
if isfield(Input, 'detrend') && Input.detrend
    opts.mean_signal=output.mean_signal;
end
tic
[timeseries_1, Varg] = incremental_temporal_update_gpu(output.forward_model_, Input.LFM_folder, [], Input.Junk_size, Input.x_offset,Input.y_offset,Input.dx,Nnum,opts);
toc
output.timeseries_total=zeros(size(timeseries_1,1),length(Varg));
output.timeseries_total(:, Varg==1) = timeseries_1;
output.timeseries_total(:, Varg==0) = output.timeseries_;
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Extraction complete']);

%% save output
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Saving result'])
output.Input = Input;
save(fullfile(Input.output_folder, Input.output_name), '-struct', 'output', '-v7.3');
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'main_nnmf_SID() returning'])
end
