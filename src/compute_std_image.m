function std_image=compute_std_image(Y,y_1,y_2)
%% COMPUTE_STD_IMAGE  Compute the standard deviation image of the difference of a movie Y and a tensor product.
%
% Input: 
% Y                     2D-array of size (M,N)
% y_1                   M x 1 array
% y_2                   1 x N array
%
% Output:
% std_image             Standard deviation image

if nargin<2
    y_1=zeros(size(Y,1),1);
    y_2=zeros(1,size(Y,2));
end

A = sum(Y.^2,2);
B = mean(Y,2);
C = mean(y_2);

std_image=sqrt((A - y_1.^2)/length(y_2) - B.^2 + 2*B.*y_1*C - y_1.^2*C^2);
end

