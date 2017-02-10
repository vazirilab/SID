function Backprojection = backwardProjectACC(Ht, projection, CAindex )
barshow=0;
x3length = size(Ht,5);
Nnum = size(Ht,3);
Backprojection = zeros(size(projection, 1), size(projection, 2), x3length);
zeroSlice = zeros(  size(projection,1) , size(projection, 2));

if barshow,
    k = 0;
    wb = waitbar(0,'Please wait...backward projecting');
end


for cc=1:x3length,
    tempSliceBack = zeroSlice;
    for aa=1:Nnum,
        for bb=1:Nnum,        
            if barshow,
                k = k + 1;
                waitbar(k/(Nnum*Nnum*x3length),wb);
            end       
            
            Hts = squeeze(Ht( CAindex(cc,1):CAindex(cc,2), CAindex(cc,1):CAindex(cc,2) ,aa,bb,cc));            
            tempSlice = zeroSlice;
            tempSlice( (aa:Nnum:end) , (bb:Nnum:end) ) = projection( (aa:Nnum:end) , (bb:Nnum:end) );
            tempSliceBack = tempSliceBack + conv2(tempSlice, Hts, 'same');   

        end
    end
    Backprojection(:,:,cc) = Backprojection(:,:,cc) + tempSliceBack;
end

if barshow,
    delete(wb);
end
