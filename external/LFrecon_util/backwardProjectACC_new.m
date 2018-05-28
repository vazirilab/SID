function Backprojection = backwardProjectACC_new(H, projection, CAindex )

barshow=0;
x3length = size(H,5);
Nnum = size(H,3);
Backprojection = zeros(size(projection, 1), size(projection, 2), x3length);

if barshow,
    k = 0;
    wb = waitbar(0,'Please wait...backward projecting');
end


for cc=1:x3length,
    for aa=1:Nnum,
        for bb=1:Nnum,        
            if barshow,
                k = k + 1;
                waitbar(k/(Nnum*Nnum*x3length),wb);
            end 
            filter=imrotate(squeeze(H( CAindex(cc,1):CAindex(cc,2), CAindex(cc,1):CAindex(cc,2) ,aa,bb,cc)),180);
            temp=conv2(projection, filter,'same');
            Backprojection((aa:Nnum:end),(bb:Nnum:end),cc)=temp((aa:Nnum:end),(bb:Nnum:end));
        end
    end
end

if barshow,
    delete(wb);
end
end