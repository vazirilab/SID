function centers = segment_component(component, threshold)

    component=component/max(component(:));
    component(isnan(component))=0;
    component=component-threshold;
    component(component<0)=0;
    centers=[];
    
    B=reshape(component,[],1);
    beads=bwconncomp((component));
    for k=1:beads.NumObjects
        qu=B(beads.PixelIdxList{1,k});
        q=sum(B(beads.PixelIdxList{1,k}));
        [a,b,c]=ind2sub(size(component),beads.PixelIdxList{1,k});
        centers(k,:)=([a,b,c]'*qu/q)';
    end
    
end