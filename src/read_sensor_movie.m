function [sensor_movie, movie_size] = read_sensor_movie(in_folder, x_offset, y_offset, dx, Nnum, do_rectify, frames, mask, crop_border_microlenses)
% READ_SENSOR_MOVIE: Algorithm loads the frames, specified in the struct 
% "frames" of the movie contained in the folder "in_folder" into the 
% working memory.
%
% Input:
% x_offset, y_offset, dx... Lenslet-parameters for rectification.
% Nnum...                   number of pixels behind microlens.
% do_rectification...       boolean that determines wether the raw movie
%                           frames ought to be rectified.
% struct frames:
%   frames.start...         First frame to be loaded.
%   frames.step...          algorithm only loads frames with increments
%                           of 'step' between them.
%   frames.end...           Final frame to be loaded.
%   frames.mean...          boolean that determines wether to load just the
%                           frames specified by the struct "frames" or if 
%                           the algorithm loads all frames, and
%                           computes the mean of the junks of frames in 
%                           between the frames specified by the struct 
%                           'frames'.
%   frames.list             Vector of frame indices to be loaded by
%                           algorithm; if this field exists frames.start, 
%                           frames.step,frames.end is ignored.
%
% Output:
% sensor_movie...           Resulting framewise linearized movie
% num_frames_total...       dimensions of the movie in 'in_folder'

if nargin < 8
    mask = true;
end

if nargin < 9
    crop_border_microlenses = [0 0 0 0];
end

if exist(in_folder, 'dir')
    infiles_struct = dir(fullfile(in_folder, '/*.tif*'));
    [~, order] = sort({infiles_struct(:).name});
    infiles_struct = infiles_struct(order);
else
    infiles_struct = dir(fullfile(in_folder));
    [in_folder, ~, ~] = fileparts(in_folder);
    disp('LFM_folder does not exist');
end

if nargin < 7 || isempty(frames)
    frames.start = 1;
    frames.step = 1;
    frames.end = size(infiles_struct, 1);
    frames.mean = 0; % disable moving average
end

if isfield(frames,'list') && ~isempty(frames.list)
    frames_=frames.list(frames<=size(infiles_struct,1));
else
    frames_=frames.start:frames.step:min(frames.end,size(infiles_struct,1));
end

movie_size(3)=size(infiles_struct,1);
if frames.mean
    for frame=1:length(frames_)-1        
        for frame_=[frames_(frame):frames_(frame+1)]
            if do_rectify
                img_rect = ImageRect(...
                    double(imread(fullfile(in_folder, infiles_struct(frame_).name), 'tiff')), ...
                    x_offset, y_offset, dx, Nnum, true, ...
                    crop_border_microlenses(1), crop_border_microlenses(2), crop_border_microlenses(3), crop_border_microlenses(4));
            else
                img_rect = single(imread(fullfile(in_folder, infiles_struct(frame_).name), 'tiff'));
                img_rect = img_rect(crop_border_microlenses(3)*Nnum + 1 : ...
                    end - crop_border_microlenses(4)*Nnum, crop_border_microlenses(1)*Nnum + 1, ...
                    end - crop_border_microlenses(2)*Nnum);
            end
            if frame_==frames_(frame)
                sens = ones(numel(img_rect),length([frames_(frame):frames_(frame+1)]) , 'single');
                if frame==1
                    sensor_movie=ones(numel(img_rect), length(frames_)-1, 'single');
                    movie_size(1:2)=size(img_rect);
                end
            end
            sens(:,frame_-frames_(frame)+1) = img_rect(:);

        end
        if mod(frame, 20) == 1
            fprintf([num2str(frames_(frame)) ' ']);
        end
        sensor_movie(:,frame)=mean(sens,2);
    end
else
    infiles_struct = infiles_struct(frames_);
    for frame = 1:size(infiles_struct,1)
        if mod(frame, 20) == 1
            fprintf([num2str(frame) ' ']);
        end
        if do_rectify
            img_rect = ImageRect(double(imread(fullfile(in_folder, ...
                infiles_struct(frame).name), 'tiff')) .* mask, x_offset,...
                y_offset, dx, Nnum, true, crop_border_microlenses(3), ...
                crop_border_microlenses(4), crop_border_microlenses(1),...
                crop_border_microlenses(2));
        else
            img_rect = single(imread(fullfile(in_folder, infiles_struct(frame).name), ...
                'tiff')) .* mask;
            img_rect = img_rect(crop_border_microlenses(1)*Nnum + 1 : end -...
                crop_border_microlenses(2)*Nnum, crop_border_microlenses(3)*Nnum ...
                + 1 : end - crop_border_microlenses(4)*Nnum);
        end
        if frame == 1
            %         sensor_movie = ones(size(img_rect, 1), size(img_rect, 2), ...
            %           size(infiles_struct,1), 'double');
            sensor_movie = ones(numel(img_rect), size(infiles_struct,1), 'single');
            movie_size(1:2)=size(img_rect);
        end
        if size(infiles_struct)==1
            %         sensor_movie(:, :) = img_rect;
            sensor_movie= img_rect(:);
        else
            %         sensor_movie(:, :, img_ix) = img_rect;
            sensor_movie(:, frame) = img_rect(:);
        end
    end
end
fprintf('\n');
end
