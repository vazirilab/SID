function Ifiltered=band_pass_filter(Image, up, down, id,schief)
tic

if id>0
    gpu = gpuDevice(id);
end

Ifiltered = Image/max(Image(:));
for iter=1:up-down
    Ifiltered=cellfilter(Ifiltered,up-iter,schief,id);
    Ifiltered = Ifiltered/max(Ifiltered(:));
    Ifiltered(Ifiltered<0) = 0;
end

if id>0
    gpu.reset()
    clear gpu
    gpuDevice([]);
end
toc
end