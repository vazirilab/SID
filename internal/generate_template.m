function template=generate_template(centers,H,std_image,thres)

if nargin<4
    thres=0.03;
end
disp('Generate template');

radius=squeeze(mean(mean(H,3),4));
for k=1:size(radius,3)
    radius(:,:,k)=radius(:,:,k)/max(reshape(radius(:,:,k),1,[]));
end
radius=sum(squeeze(mean(radius>thres,2))>0,1)/2;


[X,Y]=meshgrid(1:size(std_image,2),1:size(std_image,1));

template=false([size(centers,1),size(std_image)]);
for k=1:size(centers,1)
    tic
    template(k,:) = reshape(((Y-centers(k,1)).^2 + (X-centers(k,2)).^2 )<...
        radius(round(centers(k,3)))^2,1,[]);
    toc
end
disp([num2str(size(centers,1)) ' templates generated']);

end