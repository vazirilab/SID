function Xguess=reconstruction_cpu_sparse(psf_ballistic,infile,opts)

if ~isfield(opts,'maxIter')
    opts.maxIter=8;
end

forwardFUN_ =  @(Volume) forwardProjectACC( psf_ballistic.H, Volume, psf_ballistic.CAindex );
backwardFUN_ = @(projection) backwardProjectACC_new(psf_ballistic.H, projection, psf_ballistic.CAindex);
%% generate kernel

if ~isfield(opts,'rad')
    opts.rad=[2,2];
end

W=zeros(2*opts.rad(1)+1,2*opts.rad(1)+1,2*opts.rad(2)+1);
for ii=1:2*opts.rad(1)+1
    for jj=1:2*opts.rad(1)+1
        for kk=1:2*opts.rad(2)+1
            if  ((ii-(opts.rad(1)+1))^2/opts.rad(1)^2+(jj-(opts.rad(1)+1))^2/opts.rad(1)^2+(kk-(opts.rad(2)+1))^2/opts.rad(2)^2)<=1
                W(ii,jj,kk)=1;
            end
        end
    end
end
kernel=W;

forwardFUN = @(Xguess) forwardFUN_(convn(Xguess,kernel,'same'));
backwardFUN = @(projection) convn(backwardFUN_(projection),kernel,'same');

%% Reconstruction
disp('Backprojecting...');
Htf = backwardFUN(single(infile.LFmovie));
disp('Backprojection completed');
disp('Reconstructing...')
Xguess = fast_deconv(forwardFUN, backwardFUN, Htf, opts.maxIter, opts);
disp('Reconstruction completed')

end