function backprojection = backwardProjectGPU_new(H, projection)

if isa(H, 'matlab.io.MatFile')
    psf_size = size(H, 'H');
    preload = true;
else
    psf_size = size(H);
    preload = false;
end
Nnum = psf_size(3);

backprojection = gpuArray.zeros(size(projection, 1), size(projection, 2), psf_size(5), 'single');
projection = gpuArray(projection);

for cc = 1 : psf_size(5)
    if preload
        disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Loading PSF slice ' num2str(cc)]);
        H_slice = gpuArray(single(H.H(:, :, :, :, cc)));
        disp([datestr(now, 'YYYY-mm-dd HH:MM:SS') ': Done loading PSF slice ' num2str(cc)]);
    end
    for aa = 1 : Nnum
        for bb = 1 : Nnum
            if preload
                filter = rot90(H_slice(:, :, aa, bb), 2);
            else
                filter = rot90(gpuarray(single(squeeze(H(:, :, aa, bb, cc)))), 2);
            end
            temp = conv2FFT(projection, filter);
            backprojection(aa:Nnum:end, bb:Nnum:end, cc) = temp(aa:Nnum:end, bb:Nnum:end);
        end
    end
end

end