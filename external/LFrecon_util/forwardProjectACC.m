function TOTALprojection = forwardProjectACC( H, realspace, CAindex)

barshow=0;
Nnum = size(H,3);
zerospace = zeros(  size(realspace,1),   size(realspace,2), 'single');
TOTALprojection = zerospace;


if barshow,
    k = 0;
    wb = waitbar(0,'Please wait...forward projecting');
end

for aa=1:Nnum,
    for bb=1:Nnum,
        for cc=1:size(realspace,3),
            if barshow,
                k = k + 1;
                waitbar(k/(Nnum*Nnum*size(realspace,3)),wb);
            end
            
            Hs = squeeze(H( CAindex(cc,1):CAindex(cc,2), CAindex(cc,1):CAindex(cc,2) ,aa,bb,cc));              
            tempspace = zerospace;
            tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end), cc);
            projection = conv2(tempspace, Hs, 'same');
            TOTALprojection = TOTALprojection + projection;            
        end
    end
end

if barshow,
    delete(wb);
end
