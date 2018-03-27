function [timeseries,forward_model,template,indices_in_orig] = spatial_SID_update(...
    Y,timeseries,forward_model,template,indices_in_orig,opts)

if nargin<6
    opts=struct;
end

if ~isfield(opts,'tolerance')
    opts.tolerance = 1e-7;
end

if ~isfield(opts,'display')
    opts.display=false;
end

id2=[];
if opts.display
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Pruning neurons']);
end

for k=1:size(forward_model,1)
    trace=timeseries(k,:)>opts.tolerance;
    if sum(trace)>1
        id2=[id2 k];
    end
end
indices_in_orig=indices_in_orig(id2);
timeseries=timeseries(id2,:);
template=template(id2(1:end-opts.bg_sub),:);
tic
disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Starting spatial update']);
timeseries=(1./(sqrt(sum(timeseries.^2,2)))).*timeseries;
forward_model=update_spatial_component(timeseries, Y, template, opts);
if opts.display
    toc
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Spatial update completed']);
end

end