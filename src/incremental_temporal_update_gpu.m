function [timeseries, Varg]=incremental_temporal_update_gpu(forward_model, indir, bg_spatial, Junk_size, x_offset,y_offset,dx,Nnum,opts)

if exist(indir, 'dir')
    infiles_struct = dir(fullfile(indir, '/*.tif*'));
    [~, order] = sort({infiles_struct(:).name});
    infiles_struct = infiles_struct(order);
else
    disp('indir does not exist');
    return
end

if ~isfield(opts,'non_neg_on')
    opts.non_neg_on=false;
end

if ~isfield(opts,'do_crop')
    opts.do_crop=0;
end

if ~isfield(opts,'gpu_id')
    opts.gpu_id=[];
end
%% making sure what has already been computed does not get computed again
Varg=ones(1,length(infiles_struct));
if isfield(opts,'frame')
    opts.frame.end=min(opts.frame.end,length(infiles_struct));
    Varg(opts.frame.start:opts.frame.step:opts.frame.end)=0;
    line=find(Varg);
    infiles_struct=infiles_struct(logical(Varg));
else
    line=1:length(infiles_struct);
end

if isfield(opts,'baseline')
    if ~isempty(opts.baseline)
        baseline = interp1(find(~Varg)',opts.baseline,line','linear','extrap');
    else
        baseline=[1:length(infiles_struct)]*0+1;
    end
else
    baseline=[1:length(infiles_struct)]*0+1;
end
%%

num=length(infiles_struct);
timeseries=zeros(size(forward_model,1)+(~isempty(bg_spatial)),num);
mig=1:min(Junk_size,num);
flag=1;
if opts.non_neg_on
forward_model=forward_model';
end

while num>0
    for img_ix =mig
        if mod(line(img_ix), 20) == 1
            fprintf([num2str(line(img_ix)) ' ']);
        end
        img_rect =  ImageRect(double(imread(fullfile(indir, infiles_struct(img_ix).name), 'tiff')), x_offset, y_offset, dx, Nnum,0);
        if opts.do_crop
            img_rect = img_rect(opts.crop.x_min+1:opts.crop.x_max,opts.crop.y_min+1:opts.crop.y_max);
        end
        if img_ix == mig(1)
            sensor_movie = ones(size(img_rect, 1)*size(img_rect, 2), length(mig), 'double');
        end
        if size(infiles_struct)==1
            sensor_movie(:) = img_rect(:)/baseline(img_ix);
        else
            sensor_movie(:, img_ix-mig(1)+1) = img_rect(:)/baseline(img_ix);
        end
    end
    if isfield(opts,'idx')
        sensor_movie=sensor_movie(opts.idx,:);
    end
    
    
    if flag==1
        if opts.non_neg_on
            [timeseries(:,mig),Q,~,h]=LS_nnls(forward_model,sensor_movie,opts);
        else
            if ~isempty(opts.gpu_id)
                forward_model = gpuArray(full(forward_model));
            end
            Q=inv(forward_model*forward_model');
            Q=Q*forward_model;
            clear forward_model;
            if ~isempty(opts.gpu_id)
                sensor_movie = gpuArray(sensor_movie);
            end
            timeseries(:,mig) = gather(Q*sensor_movie);
        end
        flag=0;
    else
        if opts.non_neg_on
            [timeseries(:,mig)]=LS_nnls(forward_model,sensor_movie,opts,Q,[],h);
        else
            if ~isempty(opts.gpu_id)
                sensor_movie = gpuArray(sensor_movie);
            end
            timeseries(:,mig) = gather(Q*sensor_movie);
        end
    end
    if isfield(opts, 'outfile')
        save(opts.outfile, 'timeseries','-v7.3');
    end
    num=num-length(mig);
    mig=(mig(length(mig)))+1:(mig(length(mig))+min(Junk_size, length(infiles_struct)-mig(length(mig))));
end

fprintf('\n');

if isfield(opts,'gpu')
    if strcmp(opts.gpu,'on')
        gpuDevice([]);
    end
end
end