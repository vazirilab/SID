function Xguess=reconstruction_cpu_sparse(psf_ballistic,infile,options)

if ~isfield(options,'maxIter')
    options.maxIter=8;
end

forwardFUN_ =  @(Volume) forwardProjectACC( psf_ballistic.H, Volume, psf_ballistic.CAindex );
backwardFUN_ = @(projection) backwardProjectACC_new(psf_ballistic.H, projection, psf_ballistic.CAindex);
%% generate kernel
if ~isfield(options,'form')
    options.form='spherical';
end

if isfield(options,'rad')
    if strcmp(options.form,'spherical')
        W=zeros(2*options.rad(1)+1,2*options.rad(1)+1,2*options.rad(2)+1);
        for ii=1:2*options.rad(1)+1
            for jj=1:2*options.rad(1)+1
                for kk=1:2*options.rad(2)+1
                    if  ((ii-(options.rad(1)+1))^2/options.rad(1)^2+(jj-(options.rad(1)+1))^2/options.rad(1)^2+(kk-(options.rad(2)+1))^2/options.rad(2)^2)<=1
                        W(ii,jj,kk)=1;
                    end
                end
            end
        end
    elseif strcmp(options.form,'gaussian')
        gaussian=fspecial('gaussian',ceil(10*options.rad(1))+1,options.rad(1));
        W=reshape(reshape(gaussian,[],1)*exp(-[-3*options.rad(2):1:3*options.rad(2)].^2/4/options.rad(2)^2),ceil(10*options.rad(1))+1,ceil(10*options.rad(1))+1,[]);
        
    elseif strcmp(options.form,'lorentz')
        [X Y Z] = meshgrid([-ceil(5*options.rad(1)):ceil(5*options.rad(1))],[-ceil(5*options.rad(1)):ceil(5*options.rad(1))],[-ceil(5*options.rad(2)):ceil(5*options.rad(2))]);
        W = 1./(1 + (options.rad(1)*(X.^2 + Y.^2) + options.rad(2)*Z.^2));    
    end
    W = W/norm(W(:));
    kernel=gpuArray(W);
    
    forwardFUN = @(Xguess) forwardFUN_(convn(Xguess,kernel,'same'));
    backwardFUN = @(projection) convn(backwardFUN_(projection),kernel,'same');
    
else
    forwardFUN = forwardFUN_;
    backwardFUN = backwardFUN_;
end


%% Reconstruction
disp('Backprojecting...');
Htf = backwardFUN(single(infile.LFmovie));
disp('Backprojection completed');
disp('Reconstructing...')
Xguess = fast_deconv(forwardFUN, backwardFUN, Htf, options.maxIter, options);
disp('Reconstruction completed')

end