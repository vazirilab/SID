function TOTALprojection = forwardProjectGPU(H, realspace)

if isa(H, 'matlab.io.MatFile')
    psf_size = size(H, 'H');
    preload = true;
else
    psf_size = size(H);
    preload = false;
end
Nnum = psf_size(3);

zerospace = gpuArray.zeros(size(realspace,1), size(realspace,2), 'single');
TOTALprojection = zerospace;

for cc = 1 : size(realspace, 3)
    if preload
        H_slice = gpuArray(single(H.H(:, :, :, :, cc)));
    end
    for aa = 1 : Nnum
        for bb = 1 : Nnum
            if preload
                Hs = H_slice(:, :, aa, bb);
            else
                Hs = gpuArray(single(squeeze(H(:, :, aa, bb, cc))));
            end
            tempspace = zerospace;
            tempspace(aa:Nnum:end, bb:Nnum:end) = realspace(aa:Nnum:end, bb:Nnum:end, cc);
            projection = conv2FFT(tempspace, Hs);
%             projection = conv_fft2_v2(tempspace, Hs, 'same');
%             projection = conv2(tempspace, Hs, 'same');
            TOTALprojection = TOTALprojection + projection;            
        end
    end
end

