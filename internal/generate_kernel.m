function kernel = generate_kernel(ker_shape,ker_param)

disp('Generating kernel');

if nargin<2
    if strcmp(ker_shape,'ball')
        [X,Y,Z] = meshgrid(-ker_param(1):ker_param(1),-...
            ker_param(1):ker_param(1),-ker_param(2):ker_param(2));
        kernel = single((X.^2/ker_param(1)^2 + Y.^2/ker_param(1)^2 +...
            Z.^2/ker_param(2)^2)<=1);
    elseif strcmp(ker_shape,'gaussian')
        gaussian=fspecial('gaussian',ceil(10*ker_param(1))+1,...
            ker_param(1));
        kernel=reshape(reshape(gaussian,[],1)*exp(-[-3*ker_param(2):...
            1:3*ker_param(2)].^2/4/ker_param(2)^2),ceil(...
            10*ker_param(1))+1,ceil(10*ker_param(1))+1,[]);
    elseif strcmp(ker_shape,'lorentz')
        [X,Y,Z] = meshgrid([-ceil(5*ker_param(1)):ceil(...
            5*ker_param(1))],[-ceil(5*ker_param(1)):...
            ceil(5*ker_param(1))],[-ceil(5*ker_param(2)):...
            ceil(5*ker_param(2))]);
        kernel = 1./(1 + (ker_param(1)*(X.^2 + Y.^2) + ker_param(2)*Z.^2));
    elseif strcmp(ker_shape,'user')
        kernel=ker_param;
    end
    kernel = kernel/norm(kernel(:));
else
    kernel = [];
end
disp('Prepared Reconstruction kernel');

end