function mean_signal=par_mean_signal(indir,step, x_offset,y_offset,dx,Nnum,prime)

if nargin==2
    step=1;
end

%%
if exist(indir, 'dir')
    infiles_struct = dir(fullfile(indir, '/*.tif*'));
    [~, order] = sort({infiles_struct(:).name});
    infiles_struct = infiles_struct(order);
else
    disp('indir does not exist');
    return
end

% disp('Input files:');
% for i=1:size(infiles_struct)
%     disp(infiles_struct(i));
% end

%%
FileTif=fullfile(indir, infiles_struct(1).name);
InfoImage=imfinfo(FileTif);
mImage=InfoImage(1).Width;
nImage=InfoImage(1).Height;
NumberImages=length(InfoImage);
bg_spatial=ones(nImage,mImage,NumberImages,'double');

par_C = gcp('nocreate');

if isempty(par_C)
    par_C=parpool;
end

if nargin<8
    prime=size(infiles_struct,1);
end

prime=min(prime,size(infiles_struct,1));
n_=length(infiles_struct);
infiles_struct = infiles_struct(1:step:prime);
N=par_C.NumWorkers;

mean_signal_par=zeros([par_C.NumWorkers,ceil(length(infiles_struct)/par_C.NumWorkers)]);
parfor worker=1:par_C.NumWorkers
    for i=1:length(worker:N:length(infiles_struct))
        img_rect = ImageRect(double(imread(fullfile(indir, infiles_struct(worker+(i-1)*N).name), 'tiff')), x_offset, y_offset, dx, Nnum,0);
        parvar=zeros(size(squeeze(mean_signal_par(worker,:))));
        parvar(i)=mean(mean(img_rect));
        mean_signal_par(worker,:)=mean_signal_par(worker,:)+parvar;
    end
end

for worker=1:par_C.NumWorkers
    mean_signal(worker:N:length(infiles_struct))=mean_signal_par(worker,1:length(worker:N:length(infiles_struct)))';
end

mean_signal=interp1([1:step:prime],mean_signal,[1:n_]);

end
