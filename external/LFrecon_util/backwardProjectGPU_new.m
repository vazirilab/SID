function Backprojection = backwardProjectGPU_new(H, projection)

Nnum = size(H,3);
x3length = size(H,5);
Backprojection = gpuArray.zeros(size(projection, 1), size(projection, 2), x3length , 'single');
projection=gpuArray(projection);

Xsize = [size(projection,1),size(projection,2)];
msize = [size(H,1), size(H,2)];
mmid = floor(msize/2);
Xsize = Xsize + mmid;
Xsize = [ min( 2^ceil(log2(Xsize(1))), 256*ceil(Xsize(1)/256) ), min( 2^ceil(log2(Xsize(2))), 256*ceil(Xsize(2)/256) ) ];
zeroImageX = gpuArray(zeros(Xsize, 'single'));


for cc=1:x3length
    for aa=1:Nnum
        for bb=1:Nnum                  
            filter = gpuArray(imrotate(squeeze(H(:,:, aa,bb,cc)),180)); 
            temp = conv2FFT_(projection, filter,zeroImageX,Xsize);

%             temp=conv2FFT(projection, filter);
            Backprojection((aa:Nnum:end),(bb:Nnum:end),cc)=temp((aa:Nnum:end),(bb:Nnum:end));
        end
    end
end

end
