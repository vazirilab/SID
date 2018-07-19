function [forward_model, timeseries, template, indices_in_orig] = ...
    merge_filters(forward_model, timeseries, template, indices_in_orig, opts)
% MERGE_FILTERS merges components of the SID-nnmf, that overlap by more than 30% in
% their templates and have a correlation of more than opts.limit in their temporal signals (default: 0.9)

if nargin<5
    opts=struct;
end

if ~isfield(opts,'limit')
    opts.limit=0.90;
end

if ~isfield(opts,'bg_sub')
    opts.bg_sub=true;
end

corr_m = corrcoef(timeseries')-eye(size(timeseries,1));
corr_m = corr_m(1:end-opts.bg_sub,1:end-opts.bg_sub);


temp = single(template);
temp =(1./min(sum(temp,2),sum(temp,2)')).*( temp*temp')-eye(size(timeseries,1)-opts.bg_sub);


temp = single(temp>1/3) + corr_m;

temp = temp(1:end-opts.bg_sub,1:end-opts.bg_sub);

temp = tril(temp>1+opts.limit);

for ii=find(sum(temp,1))
    template(ii,:)=(template(ii,:)+sum(template(temp(:,ii),:),1))>0;
    template(temp(:,ii),:)=0;
end

if opts.bg_sub
    timeseries = timeseries([find(sum(template,2)>0)' size(timeseries,1)],:);
    forward_model = forward_model([find(sum(template,2)>0)' size(timeseries,1)],:);
    indices_in_orig = indices_in_orig([find(sum(template,2)>0)' size(timeseries,1)]);
else
    timeseries = timeseries(sum(template,2)>0,:);
    forward_model = forward_model(sum(template,2)>0,:);
    indices_in_orig = indices_in_orig(sum(template,2)>0);
end


template = template(sum(template,2)>0,:);
end


