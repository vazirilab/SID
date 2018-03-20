function    [Y,output] = crop(Y,output,Inside,crop_mask,Nnum)

mx=mean(crop_mask,2);
my=mean(crop_mask,1);
output.crop.x_min=min(find(mx));
output.crop.y_min=min(find(my));
output.crop.x_max=max(find(mx));
output.crop.y_max=max(find(my));
output.crop.x_min=floor(output.crop.x_min/Nnum)*Nnum;
output.crop.y_min=floor(output.crop.y_min/Nnum)*Nnum;
output.crop.x_max=ceil(output.crop.x_max/Nnum)*Nnum;
output.crop.y_max=ceil(output.crop.y_max/Nnum)*Nnum;
Y=reshape(Y,size(output.std_image,1),size(output.std_image,2),[]);
Y=Y(output.crop.x_min+1:output.crop.x_max,output.crop.y_min+1:output.crop.y_max,:);
output.microlenses=output.microlenses(output.crop.x_min+1:output.crop.x_max,output.crop.y_min+1:output.crop.y_max,:);
output.bg_spatial=output.bg_spatial(output.crop.x_min+1:output.crop.x_max,output.crop.y_min+1:output.crop.y_max,:);
Inside=Inside(output.crop.x_min+1:output.crop.x_max,output.crop.y_min+1:output.crop.y_max,:);
output.std_image=output.std_image(output.crop.x_min+1:output.crop.x_max,output.crop.y_min+1:output.crop.y_max,:);
Y=reshape(Y,size(output.std_image,1)*size(output.std_image,2),[]);
output.idx=find(Inside>0);
output.movie_size(1:2) = size(output.std_image);

end