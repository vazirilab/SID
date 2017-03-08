
function [std_image, mean_image]=par_compute_std_image(indir, step, bg_temporal, bg_spatial, prime, x_offset,y_offset,dx,Nnum)

if nargin==2
    step=1;
elseif nargin<4
    bg_temporal=[];
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
img=zeros(nImage,mImage,NumberImages,'double');

if ~isempty(bg_temporal)
    bg_spatial=bg_spatial(:)';
end

par_C = gcp('nocreate');

if isempty(par_C)
    par_C=parpool;
end

if max((nargin<6),isempty(prime))
    prime=size(infiles_struct,1);
end

prime=min(prime,size(infiles_struct,1));


infiles_struct = infiles_struct(1:step:prime);
N=par_C.NumWorkers;
std_image= zeros([par_C.NumWorkers,size(img(:))]);
mean_image = std_image;

parfor worker=1:par_C.NumWorkers
    for i=worker:N:length(infiles_struct)
        if NumberImages==1
            img_rect = double(imread(fullfile(indir, infiles_struct(i).name), 'tiff'));
        else
            for ii=1:NumberImages
                img_rect(:,:,ii)=double(imread(FileTif,'Index',ii));
            end
        end
        if ~isempty(bg_temporal)
            img_rect=img_rect(:)'-bg_spatial*bg_temporal(i);
        else
            img_rect=img_rect(:)';
        end
        A = img_rect(:) - squeeze(mean_image(worker,:))';
        mean_image(worker,:)=squeeze(mean_image(worker,:))+(img_rect(:)'-squeeze(mean_image(worker,:)))/i;
        A = A .* (img_rect(:)'-squeeze(mean_image(worker,:)))';
        std_image(worker,:) = squeeze(std_image(worker,:))' + A;
        fprintf([num2str(i) ' ']);
    end
end

fprintf('\n')

xa=squeeze(mean_image(1,:));
Ma=squeeze(std_image(1,:));
na=length(1:N:length(infiles_struct));

for worker=2:par_C.NumWorkers
    A=squeeze(mean_image(worker,:))-xa;
    xa=(xa*na+squeeze(mean_image(worker,:))*length(worker:N:length(infiles_struct)))/(na+length(worker:N:length(infiles_struct)));
    Ma=Ma+squeeze(std_image(worker,:))+(A.^2)*na*length(worker:N:length(infiles_struct))/(na+length(worker:N:length(infiles_struct)));
    na=na+length(worker:N:length(infiles_struct));
end


std_image = reshape(Ma/(length(infiles_struct)-1),size(img));
xa=reshape(xa,size(img));

if nargin>6
    
    std_image =  sqrt(ImageRect(std_image, x_offset, y_offset, dx, Nnum,0));
    xa=ImageRect(xa, x_offset, y_offset, dx, Nnum,0);
else
    std_image=sqrt(std_image);
end
mean_image=xa;
end
