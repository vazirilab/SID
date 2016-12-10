%% fish timeseries extraction
%
%
%
%% Input
Input.LFM_folder='/ssd_raid_4TB/oliver.s/fish-7-10-2016/fish1_LFM_20x05NA_exc12pc/';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_olympus_20x_05NA_water__20FN__on_scientifica_from-100_to100_zspacing4_Nnum11_lambda520_OSR3_';
Input.output_folder='/ssd_raid_4TB/oliver.s/Desktop/new_method/LFM-factorization/';
Input.output_name='fish1_LFM_20x05NA_exc12pc';
Input.x_offset=1277.700000;
Input.y_offset=1082.00000;
Input.rank=30;
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
[S,T]=nnmf(sensor_movie,Input.rank);
S=S(:,squeeze(max(S,[],1))>0);
S=[S output.std_image(:)];
sensor_movie=sensor_movie(output.idx,:);
S=S';

%% reconstruct spatial filters
disp('Reconstructing spatial filters');
psf_ballistic=load(Input.psf_filename_ballistic);
poolobj = gcp('nocreate');
delete(poolobj);

nn=length(Input.gpu_ids);
gimp=Input.gpu_ids;
parpool(nn);

options_rec.p=2;
options_rec.mode='TV';
options_rec.lambda=[ 0.1, 0.1, 5];

for kk=1:nn:size(S,1)
    img=cell(nn,1);
    for worker=1:min(nn,size(S,1)-(kk-1))
        k=kk+worker-1;
        pre=S(k,:);
        img_=reshape(pre,size(output.std_image,1),[]);
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
        options{worker}.gpu_ids=mod((worker-1),nn)+1;
        options{worker}.gpu_ids=gimp(options{worker}.gpu_ids);
        recon{worker}= reconstruction_gpu_oliver(infile, psf_ballistic, '/home/oliver.skocek/Desktop/zm200', '/tmp', 16, options{worker},options_rec);
        gpuDevice([]);
    end
    for kp=1:min(nn,size(S,1)-(kk-1))
        output.recon{kk+kp-1}=recon{kp};
    end
    disp(kk)
end

%% filter reconstructed spatial filters
disp('Filtering reconstructed spatial filters');
Hsize = size(psf_ballistic.H);
m=[size(output.std_image,1),size(output.std_image,2),Hsize(5)];
bordz = 15;
bord=1;
cellSize = 25;

[X,Y,Z]=meshgrid(1:2:2*size(output.std_image,2)-1,1:2:2*size(output.std_image,1)-1,[1:Input.native_focal_plane-1 Input.native_focal_plane+1:Hsize(5)]);
[Xq,Yq,Zq]=meshgrid(1:2*size(output.std_image,2)-1,1:2*size(output.std_image,1)-1,1:Hsize(5));

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
        filtered_Image_=band_pass_filter(img{worker}, cellSize, 8, gimp(worker),1.8);
        segm_(worker,:,:,:)=filtered_Image_(bord:size(filtered_Image_,1)-bord,bord:size(filtered_Image_,2)-bord,bordz+1:bordz+Hsize(5));
        
        gpuDevice([]);
    end
    for kp=1:min(nn,size(S,1)-(kk-1))
        filtered_Image=zeros(size(Ifiltered)-[0 0 2*bordz]);
        filtered_Image(bord:size(Ifiltered,1)-bord,bord:size(Ifiltered,2)-bord,:)=squeeze(segm_(kp,:,:,:));
        segmm{kk+kp-1}=filtered_Image;
    end
    disp(kk)
end
%% generate brain model
disp('Generate brain model');
segm=zeros(size(segmm{1}));

for kk=1:size(S,1)
    segm_=segmm{kk}-0.035;
    segm_(segm_<0)=0;
    %                                                                       %newly added please evaluate first
    beads=bwconncomp(segm_);                                                %purpose is to get rid of small islands
    segm_=segm_(:);
    segm__=zeros(size(segm_));
    for k=1:beads.NumObjects
        if numel(beads.PixelIdxList{k})>20
            segm__(beads.PixelIdxList{k})=segm_(beads.PixelIdxList{k});
        end
    end
    segm_=reshape(segm__,size(segm));
    %
    segm=segm+segm_;
    disp(kk);
end

output.segm=segm;

%% extract neuron centers
disp('Extract neuronal centers from brain model');
output.centers=[];
B=reshape(segm,[],1);
beads=bwconncomp(imregionalmax(segm));
for k=1:beads.NumObjects
    qu=B(beads.PixelIdxList{1,k});
    q=sum(B(beads.PixelIdxList{1,k}));
    [a,b,c]=ind2sub(size(segm),beads.PixelIdxList{1,k});
    output.centers(k,:)=([a,b,c]'*qu/q)';
end
output.segmm=segmm;
clearvars -except sensor_movie Input output mean_signal psf_ballistic Hsize m

output.centers(:,1:2)=(output.centers(:,1:2)-1)/2+1;

%% Initiate forward_model
disp('Initiate forward_model');
forward_model_indices=cell(1,size(output.centers,1));
forward_model_values=forward_model_indices;
N=0;

rr=3;
r=4;
BW=[];
BWW=[];
W=zeros(2*r,2*r,2*rr);
for ii=1:2*r
    for jj=1:2*r
        for kk=1:2*r
            if  ((ii-((2*r-1)/2+1))^2/r^2+(jj-((2*r-1)/2+1))^2/r^2+(kk-((2*rr-1)/2+1))^2/rr^2)<=1
                W(ii,jj,kk)=1;
            end
        end
    end
end
BW=bwconncomp(W);
[BWW(:,1) BWW(:,2) BWW(:,3)]=ind2sub([2*r,2*r,2*rr],BW.PixelIdxList{1,1});

for k=1:size(output.centers,1)
    B=[];
    for j=1:size(BWW,1)
        bbb=round(BWW(j,:)-[((2*r-1)/2+1)*[1 1] ((2*rr-1)/2+1)]+output.centers(k,:));
        if (bbb(1)<=m(1))&&(bbb(1)>0)&&(bbb(2)<=m(2))&&(bbb(2)>0)&&(bbb(3)<=Hsize(5))&&(bbb(3)>0)
            B=[B' bbb']';
        end
    end
    gt{1,1}=B;
    Q=forwardproject(gt,psf_ballistic,size(output.std_image));
    Q=Q/norm(Q);
    forward_model_indices{1,k}=find(Q);
    forward_model_values{1,k}=Q(forward_model_indices{1,k});
    N=N+length(forward_model_values{1,k});
    %     disp(k);
end
I=zeros(N,1);
J=I;
S=I;
jj=0;
for k=1:size(forward_model_indices,2)
    J(jj+1:jj+size(forward_model_values{1,k},2))= forward_model_indices{1,k};
    I(jj+1:jj+size(forward_model_values{1,k},2))=k*ones(size(forward_model_values{1,k}));
    S(jj+1:jj+size(forward_model_values{1,k},2))=forward_model_values{1,k};
    jj=jj+size(forward_model_values{1,k},2);
    %     disp(k)
end
output.forward_model=sparse(I,J,S,size(output.centers,1),size(output.std_image,1)*size(output.std_image,2));
toc;
disp([num2str(neuron) ' NSFs generated']);
%% generate template
disp('Generate template');
Nnum=psf_ballistic.Nnum;
II=[];JJ=[];
tic
for neuron=1:size(output.forward_model,1)
    img=reshape(output.forward_model(neuron,:),size(output.std_image));
    img_=zeros(size(output.std_image)/Nnum);
    for k=1:size(output.std_image,1)/Nnum
        for j=1:size(output.std_image,2)/Nnum
            img_(k,j)=mean(mean(img((k-1)*Nnum+1:k*Nnum,(j-1)*Nnum+1:j*Nnum)));
        end
    end
    img_=img_/max(img_(:));
    img_(img_<0.07)=0;
    [I_,J_,~]=find(img_);
    I=[];J=[];
    for k=1:length(I_)
        s=(I_(k)-1)*Nnum+1:I_(k)*Nnum;
        for l=1:Nnum
            I=[I' (ones(Nnum,1)*s(l))']';
            J=[J' ((J_(k)-1)*Nnum+1:J_(k)*Nnum)]';
        end
    end
    II=[II' (ones(size(I))*neuron)']';
    JJ=[JJ' sub2ind(size(img),I,J)']';
    %     disp(neuron);
end
toc
template=sparse(II,JJ,ones(size(II)),size(output.forward_model,1),size(output.std_image,1)*size(output.std_image,2));
output.template=template;
toc
disp([num2str(neuron) ' templates generated']);
%% croping model

neur=find(squeeze(max(output.forward_model(:,output.idx),[],2)>0));
output.forward_model_=output.forward_model(neur,output.idx);

template_=template(neur,output.idx);
clearvars -except sensor_movie Input output mean_signal template_ neur Nnum neur


%% optimize model
disp('Start optimizing model')

tic
opts=[];
opts.tol=1e-7;
% opts.tol_=8e-5;
opts.gpu_ids=5;
% opts.sample=1000;
% opts.wait=1000;
% opts.max_iter=Input.temporal_iterations;
opts.display='on';
opts.gpu='on';
% opts.skip=100;
optz.exact=1;

bg_spatial_=average_ML(reshape(output.bg_spatial,size(output.bg_spatial)),Nnum);
bg_spatial_=bg_spatial_(output.idx);
bg_spatial_=bg_spatial_/norm(bg_spatial_(:));
output.forward_model_(end+1,:)=bg_spatial_;
% output.timeseries=NONnegLSQ_gpu(output.forward_model_',bg_spatial_,sensor_movie,[],opts);
disp('Starting Temporal update')
output.timeseries=fast_nnls(output.forward_model_',sensor_movie,opts);
disp('Temporal update completed');

output.timeseries_=output.timeseries;
toc

for iter=1:Input.num_iter
    id2=[];
    disp('Pruning neurons');
    for k=1:size(output.forward_model_,1)
        trace=output.timeseries_(k,:)>1e-12;
        if sum(trace)>1
            id2=[id2 k];
        end
        %         disp(k);
    end
    output.timeseries_=output.timeseries_(id2,:);
    template_=template_(id2(1:end-Input.bg_sub),:);
    output.centers=output.centers(id2(1:end-Input.bg_sub),:);
    output.forward_model=output.forward_model(id2(1:end-Input.bg_sub),:);
    tic
    disp('Starting Spatial update');
    output.timeseries_=diag(1./(sqrt(sum(output.timeseries_.^2,2))))*output.timeseries_;
    output.forward_model_=update_spatial_component(output.timeseries_, sensor_movie, template_, optz);
    toc
    disp('Spatial update completed')
    disp('Pruning neurons');
    if Input.bg_sub==1                                                      % perturb bg_spatial
        bg_spatial_=zeros(size(output.bg_spatial(:)));
        bg_spatial_(output.idx)=output.forward_model_(end,:);
        bg_spatial_=average_ML(reshape(bg_spatial_,size(output.bg_spatial)),Nnum);
        bg_spatial_=bg_spatial_(:);
        output.forward_model_(end,:)=bg_spatial_(output.idx);
    end
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
    output.forward_model_=diag(1./(sqrt(sum(output.forward_model_.^2,2))))*output.forward_model_;
    %     output.timeseries_=NONnegLSQ_gpu(output.forward_model_',[],sensor_movie,[],opts);
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
mean_signal=par_mean_signal(Input.LFM_folder,Input.step, Input.x_offset,Input.y_offset,Input.dx,Nnum,Input.prime);
opts.step=Input.step;
output.forward_model_=diag(1./(sqrt(sum(output.forward_model_.^2,2))))*output.forward_model_;
bg_spatial_=bg_spatial_/norm(bg_spatial_);
tic
output.timeseries_1=incremental_temporal_update_gpu(output.forward_model_, Input.LFM_folder, [], Input.Junk_size, Input.x_offset,Input.y_offset,Input.dx,Nnum,opts,output.idx,mean_signal);
toc
disp('Extraction complete');
%% save output

disp('Saving data')
if ~(exist(Input.output_folder)==7)
    mkdir(Input.output_folder);
end

save([Input.output_folder Input.output_name '.mat'],'-struct','output','-v7.3');

disp('COMPLETE!')
