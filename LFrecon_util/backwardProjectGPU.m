function Backprojection = backwardProjectGPU(Ht, projection)
global zeroImageEx;
global exsize;

Nnum = size(Ht,3);
x3length = size(Ht,5);
Backprojection = gpuArray.zeros(size(projection, 1), size(projection, 2), x3length , 'single');
zeroSlice = gpuArray.zeros(size(projection,1) , size(projection, 2) , 'single');


for cc=1:x3length,
    tempSliceBack = zeroSlice;
    for aa=1:Nnum,
        for bb=1:Nnum,                  
            Hts = gpuArray(squeeze(Ht(:,:, aa,bb,cc)));      
%             Hts = (squeeze(Ht(:,:, aa,bb,cc)));    
            tempSlice = zeroSlice;
            tempSlice( (aa:Nnum:end) , (bb:Nnum:end) ) = projection( (aa:Nnum:end) , (bb:Nnum:end) );
            tempSliceBack = tempSliceBack + conv2FFT(tempSlice, Hts);
%             tempSliceBack = tempSliceBack + conv_fft2_v2(tempSlice, Hts, 'same');
%             tempSliceBack = tempSliceBack + conv2(tempSlice, Hts, 'same');
        end
    end
    Backprojection(:,:,cc) = Backprojection(:,:,cc) + tempSliceBack;
end

