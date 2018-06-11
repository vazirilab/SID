function TOTALprojection = forwardProjectGPU(H, realspace)
% global zeroImageEx;
% global exsize;

Nnum = size(H,3);
zerospace = gpuArray.zeros(size(realspace,1), size(realspace,2), 'single');
TOTALprojection = zerospace;

xsize = [size(realspace,1),size(realspace,2)];
msize = [size(H,1), size(H,2)];
mmid = floor(msize/2);
xsize = xsize + mmid;
xsize = [ min( 2^ceil(log2(xsize(1))), 256*ceil(xsize(1)/256) ), min( 2^ceil(log2(xsize(2))), 256*ceil(xsize(2)/256) ) ];
zeroImageX = gpuArray(zeros(xsize, 'single'));

for aa=1:Nnum
    for bb=1:Nnum
        for cc=1:size(realspace,3)
            Hs = gpuArray(squeeze(H(:, :, aa, bb, cc)));    
            tempspace = zerospace;
            tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end), cc);
%             projection = conv2FFT(tempspace, Hs);
            projection = conv2FFT_(tempspace, Hs,zeroImageX,xsize);

            TOTALprojection = TOTALprojection + projection;            
        end
    end
end

