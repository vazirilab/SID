function Xguess=reconstruction_cpu_new(psf_ballistic,infile,opts)

if ~isfield(opts,'maxIter')
    opts.maxIter=8;
end

forwardFUN_ =  @(Volume) forwardProjectACC( psf_ballistic.H, Volume, psf_ballistic.CAindex );
backwardFUN_ = @(projection) backwardProjectACC_new(psf_ballistic.H, projection, psf_ballistic.CAindex);


if ~isfield(options,'rad')
    options.rad=[2,2];
end

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
kernel=gpuArray(W);

forwardFUN = @(Xguess) forwardFUN_(convn(Xguess,kernel,'same'));
backwardFUN = @(projection) convn(backwardFUN_(projection),kernel,'same');



disp('Backprojecting...');
Htf = backwardFUN(single(infile.LFmovie));
disp('Backprojection completed');
disp('Reconstructing...')
Xguess = fast_deconv(forwardFUN, backwardFUN, Htf, opts.maxIter, opts);
disp('Reconstruction completed')

end