function [std_image, mean_image] = par_compute_std_image(indir, step, bg_temporal, bg_spatial, final_frame, x_offset, y_offset, dx, Nnum, mask, crop_border_microlenses)
% PAR_CMPUTE_STD_IMAGE: Algorithm for the incremental computation of the 
% standard deviation image of the tif-movie conntained in the folderer 'indir'.
%
% Input: 
% step...                   algorithm only considers frames with increments
%                           of 'step' between them.
% final_frame...            Final frame of the movie the algorithm to
%                           considers.
% x_offset, y_offset, dx... Lenslet-parameters for rectification.
% Nnum...                   number of pixels behind microlens.
% bg_temporal,bg_spatial... Temporal and spatial component of a 
%                           rank-1-factorization of the tif-movie contained
%                           in 'indir'.
%                           If bg_temporal is not empty, the algorithm
%                           computes the standard deviation image of the
%                           residual of the movie in indir and the
%                           rank-1-factorization.
%
% Output:
% std_image...              Standard deviation image
% std_image...              Mean image

if nargin < 10
    mask = true;
end

if nargin < 2
    step = 1;
end

if nargin < 4
    bg_temporal = [];
    bg_spatial = [];
end

if nargin < 5
    final_frame = inf;
end

if nargin < 9
    x_offset = 0;
    y_offset = 0;
    dx = 0;
    Nnum = 0;
end

if nargin < 10
    mask = true;
end

if nargin < 11
    crop_border_microlenses = [0 0 0 0];
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

final_frame = min(final_frame, size(infiles_struct,1));

infiles_struct = infiles_struct(1:step:final_frame);
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
        img_rect = img_rect .* mask;
        
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

if all([x_offset y_offset dx Nnum])
    std_image = sqrt(ImageRect(std_image, x_offset, y_offset, dx, Nnum, 0));
    xa = ImageRect(xa, x_offset, y_offset, dx, Nnum, 0);
end

std_image=sqrt(std_image);
std_image = std_image(crop_border_microlenses(1)*Nnum + 1 : end - crop_border_microlenses(2)*Nnum, crop_border_microlenses(3)*Nnum + 1 : end - crop_border_microlenses(4)*Nnum);
mean_image=xa;
mean_image = mean_image(crop_border_microlenses(1)*Nnum + 1 : end - crop_border_microlenses(2)*Nnum, crop_border_microlenses(3)*Nnum + 1 : end - crop_border_microlenses(4)*Nnum);
end
