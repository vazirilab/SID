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

if Y<2
    y_1=zeros(size(Y,1),1);
    y_2=zeros(1,size(Y,2));
end

A=var(Y,1,2);
B=(y_1.^2)*var(y_2);
C=(y_1.^2-(sum(Y,2).*(y_1*mean(y_2))))/(length(y_2)-1);

std_image=sqrt(A+B-2*C);
end

