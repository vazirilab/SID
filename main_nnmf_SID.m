%% fish timeseries extraction
%
%
%
%% Input
Input.LFM_folder='/ssd_raid_4TB/oliver.s/fish-7-10-2016/fish1_LFM_20x05NA_exc12pc/';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_olympus_20x_05NA_water_117pm18mm_20FN_on_relay_stfica_pm100_from-100_to100_zspacing4_Nnum11_lambda520_OSR3_normed';
Input.output_folder='/ssd_raid_4TB/oliver.s/Desktop/new_method/LFM-factorization/';
Input.output_name='fish1_LFM_20x05NA_exc12pc';
Input.x_offset=1277.700000;
Input.y_offset=1082.00000;
Input.rank=30; % If Input.rank==0 SID classic instead of SID_nmf
Input.dx=22.7;
Input.step=1;
Input.step_=3;
Input.bg_iter=2;
Input.rectify=1;
Input.Junk_size=1000;
Input.bg_sub=1;
Input.prime=40000;
Input.prime_=4800;
Input.gpu_ids=[1 2 4];
Input.num_iter=4;
Input.native_focal_plane=26;
psf_ballistic=matfile(Input.psf_filename_ballistic);

%% Compute bg components via rank-1-factorization
disp('Computing background components');
if Input.bg_sub==1
    [output.bg_temporal,output.bg_spatial]=par_rank_1_factorization(Input.LFM_folder,Input.step, Input.bg_iter,Input.prime);
else
    output.bg_temporal=[];
    output.bg_spatial=[];
end

%% Compute standard-deviation image (std. image)
disp('Computing standard deviation image');
if Input.rectify==1
    [output.std_image,~]=par_compute_std_image(Input.LFM_folder,Input.step,output.bg_temporal,output.bg_spatial,Input.prime, Input.x_offset,Input.y_offset,Input.dx,psf_ballistic.Nnum);
else
    [output.std_image,~]=par_compute_std_image(Input.LFM_folder,Input.step,output.bg_temporal,output.bg_spatial,Input.prime);
end

if (Input.bg_sub==1)&&(Input.rectify==1)
    output.bg_spatial =  ImageRect(output.bg_spatial, Input.x_offset, Input.y_offset, Input.dx, psf_ballistic.Nnum,0);
end

%% load sensor movie and de-trend
disp('Loading LFM movie');
tic
sensor_movie=read_sensor_movie(Input.LFM_folder,Input.x_offset,Input.y_offset,Input.dx,psf_ballistic.Nnum,Input.step_,Input.rectify,Input.prime_);
toc
tic
disp('Detrending LFM movie');
baseline=mean(sensor_movie,1)';
baseline=fit([1:length(baseline)]',baseline,'exp2');
baseline=baseline.a*exp(baseline.b*[1:size(sensor_movie,2)])+baseline.c*exp(baseline.d*[1:size(sensor_movie,2)]);
sensor_movie=sensor_movie*diag(1./baseline);
toc

%% find crop space
disp('Finding crop space');
sub_image=output.std_image(ceil(0.8*size(output.std_image,1)):end,ceil(0.75*size(output.std_image,2)):end);
sub_image=output.std_image-mean(sub_image(:))-6*std(sub_image(:));
sub_image(sub_image<0)=0;

output.idx=find(sub_image>0);

%% generate NNMF
disp(['Generating rank-' num2str(Input.rank) '-factorization']);
opts.max_iter=1000;
opts.lambda=5;
output.centers=[];
ops.bg_temporal=squeeze(mean(sensor_movie,1));
[S, T]=fast_NMF(sensor_movie,Input.rank,opts);
S=[S' output.std_image(:)]';
sensor_movie=sensor_movie(output.idx,:);

%% reconstruct spatial filters
disp('Reconstructing spatial filters');
psf_ballistic=load(Input.psf_filename_ballistic);
poolobj = gcp('nocreate');
delete(poolobj);

options_rec.p=2;
options_rec.maxIter=8;
options_rec.mode='TV';
options_rec.lambda=[ 0, 0, 10];
options_rec.lambda_=0.1;

if isempty(Input.gpu_ids)    
    infile=struct;
    for k=1:size(S,1)
        img_=reshape(S(k,:),size(output.std_image,1),[]);
        img_=img_/max(img_(:));
        img_=img_-mean(mean(img_(ceil(0.8*size(output.std_image,1)):end,ceil(0.75*size(output.std_image,2)):end)));
        img_(img_<0)=0;
        infile.LFmovie=full(img_)/max(img_(:));
        output.recon{k} = reconstruction_cpu_sparse(psf_ballistic,infile,options_rec);
        disp(k);
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
            img_=img_-mean(mean(img_(ceil(0.8*size(output.std_image,1)):end,ceil(0.75*size(output.std_image,2)):end)));
            img_(img_<0)=0;
            img{worker}=full(img_)/max(img_(:));
        end
        options=cell(min(nn,size(S,1)-(kk-1)),1);
        recon=cell(min(nn,size(S,1)-(kk-1)),1);
        parfor worker=1:min(nn,size(S,1)-(kk-1))
            infile=struct;
            infile.LFmovie=(img{worker});
            options{worker}=options_rec;
            options{worker}.gpu_ids=mod((worker-1),nn)+1;
            options{worker}.gpu_ids=gimp(options{worker}.gpu_ids);
            
            recon{worker}= reconstruction_sparse(infile, psf_ballistic, options{worker});
            gpuDevice([]);
        end
        for kp=1:min(nn,size(S,1)-(kk-1))
            output.recon{kk+kp-1}=recon{kp};
        end
        disp(kk)
    end
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
    
    disp(ii);
end

segm=0*output.recon{1};
for ii=1:size(output.centers,1)
    segm(ceil(output.centers(ii,1)),ceil(output.centers(ii,2)),ceil(output.centers(ii,3)))=1;
end

clearvars -except sensor_movie Input output mean_signal psf_ballistic Hsize m

%% Initiate forward_model
psf_ballistic=load(Input.psf_filename_ballistic);

output.forward_model=generate_foward_model(output.centers,psf_ballistic,8,3,size(output.recon{1})); %replace 8 by 1 if _7r psf

%% generate template
output.template=generate_template(output.forward_model,psf_ballistic.Nnum,0.005,size(output.std_image));
%% croping model
neur=find(squeeze(max(output.forward_model(:,output.idx),[],2)>0));
output.forward_model_=output.forward_model(neur,output.idx);

template_=output.template(neur,output.idx);
Nnum=psf_ballistic.Nnum;
clearvars -except sensor_movie Input output mean_signal template_ neur Nnum neur

%% optimize model
disp('Start optimizing model')

tic
opts=[];
opts.tol=1e-3;
opts.tol_=1e-2;
opts.sample=1000;
opts.gpu_ids=4;
opts.display='on';
opts.gpu='on';
opts.max_iter=5000;
optz.solver=1;
optz.display='on';
optz.bg_sub=Input.bg_sub;
opts.lambda=0;

if Input.bg_sub
    bg_spatial_=average_ML(reshape(output.bg_spatial,size(output.bg_spatial)),Nnum);
    bg_spatial_=bg_spatial_(output.idx);
    bg_spatial_=bg_spatial_/norm(bg_spatial_(:));
    output.forward_model_(end+1,:)=bg_spatial_;
end

disp('Starting Temporal update')
output.timeseries=fast_nnls(output.forward_model_',sensor_movie,opts);
disp('Temporal update completed');

output.timeseries_=output.timeseries;
output.centers_=output.centers;
toc
opts.max_iter=10000;

for iter=1:Input.num_iter
    id2=[];
    disp('Pruning neurons');
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
    disp('Starting Spatial update');
    output.timeseries_=diag(1./(sqrt(sum(output.timeseries_.^2,2))))*output.timeseries_;
    output.forward_model_=update_spatial_component(output.timeseries_, sensor_movie, template_, optz);
    toc
   
    disp('Spatial update completed')
    
    if Input.update_template
        if iter==2
            for neuron=1:size(template_,1)
                crop=zeros(size(output.std_image));
                crop(output.idx)=template_(neuron,:);
                img=reshape(crop,size(output.std_image));
                img=conv2(img,ones(2*Nnum),'same')>0;
                img=img(:);
                template_(neuron,:)=(img(output.idx)>0.1);
                disp(neuron)
            end
        end
    end      
    disp('Pruning neurons');  
    id2=[];
    for k=1:size(output.forward_model_,1)
        trace=output.forward_model_(k,:)>1e-12;
        if sum(trace)>(Nnum^2)/3
            id2=[id2 k];
        end
        %         disp(k);
    end
    output.timeseries_=output.timeseries_(id2,:);
    output.forward_model_=output.forward_model_(id2,:);
    template_=template_(id2(1:end-Input.bg_sub),:);
    output.centers=output.centers(id2(1:end-Input.bg_sub),:);
    output.forward_model=output.forward_model(id2(1:end-Input.bg_sub),:);
    tic
%     output.forward_model_=diag(1./(sqrt(sum(output.forward_model_.^2,2))))*output.forward_model_;
    disp('Starting Temporal update');
    opts.warm_start=output.timeseries_;
    output.timeseries_=fast_nnls(output.forward_model_',sensor_movie,opts);
    disp('Temporal update completed');
    toc
    disp([num2str(iter) '. iteration completed']);
end
output.template_=template_;
output.Input=Input;
opts.warm_start=[];
clear sensor_movie;
disp('Model optimization completed');

%% extract time series at location LFM_folder
disp('Extracting Timeseries');
opts.step=Input.step;
opts.prime=Input.prime;
opts.warm_start=[];
opts.frame=Input.frames_for_model_optimization;
opts.idx=output.idx;
opts.max_iter=20000;
if Input.de_trend
    opts.mean_signal=output.mean_signal;
end
tic
[timeseries_1,Varg]=incremental_temporal_update_gpu(output.forward_model_, Input.LFM_folder, [], Input.Junk_size, Input.x_offset,Input.y_offset,Input.dx,Nnum,opts);
toc
output.timeseries_total=zeros(size(timeseries_1,1),length(Varg));
output.timeseries_total(:,find(Varg))=timeseries_1;
output.timeseries_total(:,find(~Varg))=output.timeseries_;
disp('Extraction complete');
%% save output

disp('Saving data')
if ~(exist(Input.output_folder)==7)
    mkdir(Input.output_folder);
end

save([Input.output_folder Input.output_name '007' '.mat'],'-struct','output','-v7.3');

disp('COMPLETE!')
