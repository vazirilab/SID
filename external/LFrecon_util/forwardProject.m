function TOTALprojection = forwardProject(H, realspace, Nnum, barshow)
zerospace = zeros(size(realspace, 1), size(realspace, 2));
TOTALprojection = zerospace;

if barshow,
    k = 0;
    wb = waitbar(0,'Please wait...forward projecting');
end

for aa=1:Nnum
    for bb=1:Nnum
        for cc=1:size(realspace, 3)
            if barshow
                k = k + 1;
                waitbar(k / (Nnum * Nnum * size(realspace, 3)), wb);
            end
            % ind =sub2ind([Nnum Nnum size(H, 3)], aa, bb, cc);
            tempspace = zerospace;
            tempspace((aa:Nnum:end), (bb:Nnum:end)) = realspace((aa:Nnum:end), (bb:Nnum:end), cc);
            projection = conv2(tempspace, H(:, :, aa, bb, cc), 'same');
            % projection = convnfft(squeeze(tempspace), squeeze(H(:, :, ind)), 'same', [1:2], option);
            TOTALprojection = TOTALprojection + projection;
        end
    end
end

if barshow
    delete(wb);
end
end

