function TOTALprojection = forwardProjectGPU(H, realspace)
global zeroImageEx;
global exsize;

Nnum = size(H,3);
zerospace = gpuArray.zeros(size(realspace,1), size(realspace,2), 'single');
TOTALprojection = zerospace;

for aa=1:Nnum,
    for bb=1:Nnum,
        for cc=1:size(realspace,3),
            Hs = gpuArray(squeeze(H(:, :, aa, bb, cc)));    
            tempspace = zerospace;
            tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end), cc);
            projection = conv2FFT(tempspace, Hs);
%             projection = conv_fft2_v2(tempspace, Hs, 'same');
%             projection = conv2(tempspace, Hs, 'same');
            TOTALprojection = TOTALprojection + projection;            
        end
    end
end

