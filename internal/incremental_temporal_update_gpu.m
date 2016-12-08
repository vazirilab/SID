function timeseries=incremental_temporal_update_gpu(forward_model, indir, bg_spatial, Junk_size, x_offset,y_offset,dx,Nnum,opts,idx, bg_temporal)

if exist(indir, 'dir')
    infiles_struct = dir(fullfile(indir, '/*.tif*'));
    [~, order] = sort({infiles_struct(:).name});
    infiles_struct = infiles_struct(order);
else
    disp('indir does not exist');
    return
end


%%
num=length(infiles_struct);
timeseries=zeros(size(forward_model,1)+(~isempty(bg_spatial)),num);
mig=1:min(Junk_size,num);
flag=1;
    forward_model=forward_model';

while num>0
    for img_ix =mig
        disp(num2str(img_ix));
        img_rect =  ImageRect(double(imread(fullfile(indir, infiles_struct(img_ix).name), 'tiff')), x_offset, y_offset, dx, Nnum,0);
        if ((img_ix==1)&&(nargin==11))
            bg_temporal=bg_temporal*mean(mean(img_rect))/bg_temporal(1);
            A=[([1:length(bg_temporal)].^2)',[1:length(bg_temporal)]',ones(length(bg_temporal),1)];
            c_=inv(A'*A)*A';
            A=[([1:length(infiles_struct)].^2)',[1:length(infiles_struct)]',ones(length(infiles_struct),1)];
            baseline=A*c_*bg_temporal';
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
    if nargin>=10
        sensor_movie=sensor_movie(idx,:);
    end
    if flag==1
        %         [timeseries(:,mig),B]=NONnegLSQ_gpu(forward_model,bg_spatial,sensor_movie,[],opts);
        [timeseries(:,mig),Q,~,H]=fast_nnls(forward_model,sensor_movie,opts);
        flag=0;
    else
        %         [timeseries(:,mig),B]=NONnegLSQ_gpu(forward_model,bg_spatial,sensor_movie,[],opts,B);
        [timeseries(:,mig),Q,~,H]=fast_nnls(forward_model,sensor_movie,opts,Q,[],H);
    end
    save('/tmp/timeseries.mat','timeseries');
    num=num-length(mig);
    mig=(mig(length(mig)))+1:(mig(length(mig))+min(Junk_size, length(infiles_struct)-mig(length(mig))));
end

gpuDevice([]);
end