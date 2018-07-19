function [forward_model, timeseries, template, indices_in_orig] = ...
    temporal_SID_update(Y, forward_model, timeseries, template, indices_in_orig, opts)

if nargin < 6
    opts = struct;
end

if ~isfield(opts, 'tolerance')
    opts.tolerance = 1e-12;
end

if ~isfield(opts, 'display')
    opts.display = true;
end

if opts.display
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Pruning neuron candidates with neglibible values in spatial component']);
end

id2=[];
for k=1:size(forward_model,1)
    trace = forward_model(k, opts.microlenses(opts.idx)>0) > opts.tolerance;
    if sum(trace) > 0  % (opts.Nnum^2)/3
        id2 = [id2 k]; %#ok<AGROW>
    elseif opts.display
        disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Discarding neuron candidate ' num2str(k)]);
    end
end
indices_in_orig = indices_in_orig(id2);
timeseries = timeseries(id2, :);
forward_model = forward_model(id2, :);
template = template(id2(1:end-opts.bg_sub), :);

if opts.display
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Pruning completed']);
end

if opts.display
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Starting temporal update']);
end
tic
forward_model = (1./ sqrt(sum(forward_model .^ 2, 2))) .* forward_model;
opts.warm_start = timeseries;

timeseries = fast_nnls(forward_model(:, opts.microlenses(opts.idx) > 0)', ...
                       Y(opts.microlenses(opts.idx) > 0, :), ...
                       opts);

if opts.display
    disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': ' 'Temporal update completed']);
    toc
end
end