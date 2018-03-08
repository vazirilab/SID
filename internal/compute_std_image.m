function std_image=compute_std_image(Y,bg_spatial,bg_temporal)

if Y<2
    bg_spatial=zeros(size(Y,1),1);
    bg_temporal=zeros(1,size(Y,2));
end

A=var(Y,1,2);
B=(bg_spatial.^2)*var(bg_temporal);
C=(bg_spatial.^2-(sum(Y,2).*(bg_spatial*mean(bg_temporal))))/(length(bg_temporal)-1);

std_image=sqrt(A+B-2*C);
end

