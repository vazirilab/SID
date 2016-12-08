function timeseries=update_temporal_component_junk(A,sensor_movie,opts)


poolobj = gcp('nocreate');
delete(poolobj);

if strcmp(opts.gpu,'on')
    opts.numworker=length(opts.gpu_ids);
end
parpool(opts.numworker);


H=full(A'*A);
Q=H./sqrt(diag(H)*diag(H)');
for worker=1:opts.numworker
    df{worker}(:,worker)=-diag(1./sqrt(diag(H)))*(A'*sensor_movie(:,worker:opts.numworker:size(sensor_movie,2)));
    option{worker}=opts;
    option{worker}.gpu_ids=opts.gpu_ids(worker);
end

parfor worker=1:opts.numworker
    timeseries_par{worker}=fast_nnls(A,[],option{worker},Q,df{worker},[]);
end

for worker=1:opts.numworker
    timeseries(:,1:opts.numworker:size(sensor_movie,2))=timeseries_par{worker};
end

if strcmp(opts.gpu,'on')
    timeseries=gather(diag(1./(sqrt(diag(H))))*timeseries);
else
    timeseries=diag(1./(sqrt(diag(H))))*timeseries;
end

end