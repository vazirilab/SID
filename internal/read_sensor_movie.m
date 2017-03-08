function sensor_movie=read_sensor_movie(LFM_folder,x_offset,y_offset,dx,Nnum,rect,frames)


%%
p.rect_dir = LFM_folder;

%%
if exist(p.rect_dir, 'dir')
    infiles_struct = dir(fullfile(p.rect_dir, '/*.tif*'));
    [~, order] = sort({infiles_struct(:).name});
    infiles_struct = infiles_struct(order);
else
    infiles_struct = dir(fullfile(p.rect_dir));
    [p.rect_dir, ~, ~] = fileparts(p.rect_dir);
    disp('LFM_folder does not exist');
end

%%
if nargin<7
    frames.start=1;
    frames.steps=1;
    frames.end=size(infiles_struct,1);
end
frames.end=min(frames.end,size(infiles_struct,1));

infiles_struct = infiles_struct(frames.start:frames.steps:frames.end);

%%
for img_ix = 1:size(infiles_struct,1)
    if mod(img_ix, 20) == 1
        fprintf([num2str(img_ix) ' ']);
    end
    if rect==1
        img_rect =  ImageRect(double(imread(fullfile(p.rect_dir, infiles_struct(img_ix).name), 'tiff')), x_offset, y_offset, dx, Nnum,0);
    else
        img_rect = double(imread(fullfile(p.rect_dir, infiles_struct(img_ix).name), 'tiff'));
    end
    if img_ix == 1
%         sensor_movie = ones(size(img_rect, 1), size(img_rect, 2), size(infiles_struct,1), 'double');

        sensor_movie = ones(numel(img_rect), size(infiles_struct,1), 'double');
    end
    if size(infiles_struct)==1
        %         sensor_movie(:, :) = img_rect;
        sensor_movie= img_rect(:);
    else
        %         sensor_movie(:, :, img_ix) = img_rect;
        sensor_movie(:, img_ix) = img_rect(:);
    end
end
fprintf('\n');
end
