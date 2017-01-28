function Backprojection = backwardProjectGPU_new(H, projection)

Nnum = size(H,3);
x3length = size(H,5);
Backprojection = gpuArray.zeros(size(projection, 1), size(projection, 2), x3length , 'single');
projection=gpuArray(projection);


for cc=1:x3length,
    for aa=1:Nnum,
        for bb=1:Nnum,                  
            filter = gpuArray(imrotate(squeeze(H(:,:, aa,bb,cc)),180));  
            temp=conv2FFT(projection, filter);
            Backprojection((aa:Nnum:end),(bb:Nnum:end),cc)=temp((aa:Nnum:end),(bb:Nnum:end));
        end
    end
end

end