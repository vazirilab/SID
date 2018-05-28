function std_image=compute_std_image(Y,y_1,y_2)
% COMPUTE_STD_IMAGE: Algorithm computes the standard deviation image of the
% difference of a movie Y and a tensorproduct.
%
% Input: 
% Y...                      movie
% y_1...                    y_1 is a size(Y,1) times 1 array.
% y_2...                    y_2 is a 1 times size(Y,2) array.
%
% Output:
% std_image...              Standard deviation image

if nargin<2
    y_1=zeros(size(Y,1),1);
    y_2=zeros(1,size(Y,2));
end

A = sum(Y.^2,2);
B = mean(Y,2);
C = mean(y_2);

std_image=sqrt((A - y_1.^2)/length(y_2) - B.^2 + 2*B.*y_1*C - y_1.^2*C^2);
end

