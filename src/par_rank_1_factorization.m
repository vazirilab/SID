function [bg_temporal, bg_spatial] = par_rank_1_factorization(indir, frames, max_iter, x_offset, y_offset, dx, Nnum, mask)
% PAR_RANK_1_FACTORIZATION Rank-1-matrix-factorization of the tif-movie Y
% contained in the indir folder.
%
%       Y~bg_spatial*bg_temporal
%
% Input:
% struct frames:
%   frames.start...         First frame to be loaded.
%   frames.step...          algorithm only loads frames with increments
%                           of 'step' between them.
%   frames.end...           Final frame to be loaded.
%   frames.mean...          boolean that determines wether to load just the
% max_iter...               maximum Number of Iterations
% x_offset, y_offset, dx... Lenslet-parameters for the rectification.
% Nnum...                   number of pixels behind microlens.
%
% Output:
% bg_temporal...            temporal component of the rank-1-factorization
% bg_spatial...             spatial component of the rank-1-factorization

if nargin < 2
    frames.start = 1;
    frames.step = 1;
    frames.end = inf;
end

if nargin < 3
    max_iter = 1;
end

if nargin < 4
    x_offset = 0;
    y_offset = 0;
    dx = 0;
    Nnum = 0;
end


if nargin < 8
    mask = true;
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
bg_spatial=ones(nImage,mImage,NumberImages,'double');

par_C = gcp('nocreate'); 

if isempty(par_C)
    par_C=parpool;
end

frames.end=min(frames.end,size(infiles_struct,1));

infiles_struct = infiles_struct(frames.start:frames.step:frames.end);
N=par_C.NumWorkers;

for iter=1:max_iter
    bg_temporal_par=zeros([par_C.NumWorkers,ceil(length(infiles_struct)/par_C.NumWorkers)]);
    bg_spatial=bg_spatial/sqrt(sum(sum(bg_spatial.^2)));
    parfor worker=1:par_C.NumWorkers
        bg_spatial_par(worker,:,:)=zeros(size(bg_spatial));
        for i=1:length(worker:N:length(infiles_struct))
            img_rect = double(imread(fullfile(indir, infiles_struct(worker+(i-1)*N).name), 'tiff')) .* mask;
            parvar=zeros(size(squeeze(bg_temporal_par(worker,:))));
            parvar(i)=sum(sum(bg_spatial.*img_rect));
            bg_temporal_par(worker,:)=bg_temporal_par(worker,:)+parvar;
            bg_spatial_par(worker,:,:)=squeeze(bg_spatial_par(worker,:,:))+img_rect*parvar(i);
        end
        btn(worker)=sum(sum(bg_temporal_par(worker,:).^2));
    end
    nrm=sqrt(sum(btn));
    bg_spatial=squeeze(sum(bg_spatial_par,1))/nrm;
    disp(['Finished iteration ' num2str(iter)]);
end

for worker=1:par_C.NumWorkers
    bg_temporal(worker:N:length(infiles_struct))=bg_temporal_par(worker,1:length(worker:N:length(infiles_struct)))';
end

if all([x_offset y_offset dx Nnum])
    bg_spatial = ImageRect(bg_spatial, x_offset, y_offset, dx, Nnum, 0);    
end

bg_temporal=bg_temporal/nrm;
end