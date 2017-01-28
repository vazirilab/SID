function [timeseries, Varg]=incremental_temporal_update_gpu(forward_model, indir, bg_spatial, Junk_size, x_offset,y_offset,dx,Nnum,opts)

if exist(indir, 'dir')
    infiles_struct = dir(fullfile(indir, '/*.tif*'));
    [~, order] = sort({infiles_struct(:).name});
    infiles_struct = infiles_struct(order);
else
    disp('indir does not exist');
    return
end


%% making sure what has already been computed does not get computed again
if isfield(opts,'frame')
opts.frame.end=min(opts.frame.end,length(infiles_struct));
Varg=ones(1,length(infiles_struct));
Varg(opts.frame.start:opts.frame.steps:opts.frame.end)=0;
line=find(Varg);
infiles_struct=infiles_struct(find(Varg));
else
    line=1:length(infiles_struct);
end
%%
num=length(infiles_struct);
timeseries=zeros(size(forward_model,1)+(~isempty(bg_spatial)),num);
mig=1:min(Junk_size,num);
flag=1;
    forward_model=forward_model';

while num>0
    for img_ix =mig
        disp(num2str(line(img_ix)));
        img_rect =  ImageRect(double(imread(fullfile(indir, infiles_struct(img_ix).name), 'tiff')), x_offset, y_offset, dx, Nnum,0);
        if ((img_ix==1)&&(nargin==11))
            if isfield(opts,'mean_signal')
                if ~isempty(opts.mean_field)
                    opts.mean_signal=opts.mean_signal*mean(mean(img_rect))/opts.mean_signal(1);
                    baseline=exp2fit([1:length(infiles_struct)],opts.mean_signal,1);
                    baseline=baseline(1)+baseline(2)*exp(-line/baseline(3));
                else
                    baseline=[1:length(infiles_struct)]*0+1;
                end
            else
                baseline=[1:length(infiles_struct)]*0+1;
            end
        end
        if img_ix == mig(1)
            sensor_movie = ones(size(img_rect, 1)*size(img_rect, 2), length(mig), 'double');
        end
        if size(infiles_struct)==1
            sensor_movie(:) = img_rect/baseline(img_ix);
            
        else
            sensor_movie(:, img_ix-mig(1)+1) = img_rect(:);
        end
    end
    if isfield(opts,'idx')
        sensor_movie=sensor_movie(opts.idx,:);
    end

    if flag==1
        %         [timeseries(:,mig),B]=NONnegLSQ_gpu(forward_model,bg_spatial,sensor_movie,[],opts);
        [timeseries(:,mig),Q,~,h]=fast_nnls(forward_model,sensor_movie,opts);
        flag=0;
    else
        %         [timeseries(:,mig),B]=NONnegLSQ_gpu(forward_model,bg_spatial,sensor_movie,[],opts,B);
        [timeseries(:,mig)]=fast_nnls(forward_model,sensor_movie,opts,Q,[],h);
    end
    save('/tmp/timeseries.mat','timeseries');
    num=num-length(mig);
    mig=(mig(length(mig)))+1:(mig(length(mig))+min(Junk_size, length(infiles_struct)-mig(length(mig))));
end

if isfield(opts,'gpu')
    if strcmp(opts.gpu,'on')
        gpuDevice([]);
    end
end
end