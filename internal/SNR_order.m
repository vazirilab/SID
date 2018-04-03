function n=SNR_order(traces,opts)

if nargin<2
    opts=struct;
end

if ~isfield(opts,'bg_sub')
    opts.bg_sub=1;
end

for k=1:size(traces,1)-opts.bg_sub
    trace = traces(k,:);
    hh = fft([trace flip(trace)]);
    hh(2:7)=0;
    hh(end-5:end)=0;
    hh=ifft(hh);
    trace=hh(1:length(hh)/2);
    trace=zscore(trace);
    sn = std(smooth(trace,8)'-trace);
    pp(k)=(max(smooth(trace))-min(smooth(trace)))/sn;
end
[~,n]=sort(-pp);

if opts.bg_sub
    n = [n size(traces,1)];
end

end