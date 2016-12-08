function Ifiltered=band_pass_filter(Image, up, down, id,schief)
tic
 gpu = gpuDevice(id);
   
    Ifiltered = Image/max(Image(:));
    for iter=1:up-down,
        Ifiltered=cellfilter(Ifiltered,up-iter,schief);
        Ifiltered = Ifiltered/max(Ifiltered(:));
        Ifiltered(find(Ifiltered<0)) = 0;
    end

gpu.reset()
clear gpu
gpuDevice([]);
toc
end