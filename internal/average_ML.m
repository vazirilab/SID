function aver=average_ML(LFM_picture,Nnum)

LFM_picture=LFM_picture-quantile(LFM_picture(:),0.36686);
LFM_picture(LFM_picture<0)=0;

fluoslide=load(['fluoslide_Nnum' num2str(Nnum) '.mat']);
fluoslide=fluoslide.fluo_slide;
fluoslide=fluoslide-quantile(fluoslide(:),0.15);
fluoslide(fluoslide<0)=0;

fluoslide=fluoslide(floor(((size(fluoslide,1)/2-size(LFM_picture,1)/2)/Nnum))*Nnum+1:...
    ceil(((size(fluoslide,1)/2+size(LFM_picture,1)/2)/Nnum))*Nnum,floor(((size(fluoslide,2)...
    /2-size(LFM_picture,2)/2)/Nnum))*Nnum+1:ceil(((size(fluoslide,2)/2+size(LFM_picture,2)/2)/Nnum))*Nnum);

for x=1:size(LFM_picture,1)/Nnum
    for y=1:size(LFM_picture,2)/Nnum
        aver((x-1)*Nnum+1:x*Nnum,(y-1)*Nnum+1:y*Nnum)=fluoslide((x-1)*...
            Nnum+1:x*Nnum,(y-1)*Nnum+1:y*Nnum)*mean(mean(abs(LFM_picture((x-1)*Nnum+1:...
            x*Nnum,(y-1)*Nnum+1:y*Nnum))))/mean(mean(abs(fluoslide((x-1)*Nnum+1:x*Nnum,(y-1)*Nnum+1:y*Nnum))));
    end
end

end