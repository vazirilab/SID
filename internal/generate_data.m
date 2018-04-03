function gt_struct = generate_data(psf_ballistic,vol_movie_size,num_neurons,radius,dim, Ca_decay,noiselevel, bleach_decay, border_no_neurons)


if max(mod(vol_movie_size(1:2),size(psf_ballistic.H,3)))&&(vol_movie_size(3)==size(psf_ballistic.H,5))
    disp('dimensions do not match psf_ballistic.');
    return
end

id=true(num_neurons,1);
msize = vol_movie_size(1:3);
msize(1:2)=msize(1:2)-2*border_no_neurons;
border_no_neurons = border_no_neurons*[1 1 0];
gt_struct.centers = border_no_neurons+ ceil(msize.*rand(num_neurons,3));
while sum(id)
    id=false(num_neurons,1);
    for k=1:num_neurons
        for j=1:k-1
            if norm(gt_struct.centers(k,:)-gt_struct.centers(j,:))<radius+10
                id(k)=true;
                disp(k)
                break
            else
                id(k)=false;
            end
        end
    end
    gt_struct.centers(id,:) = border_no_neurons+ ceil(msize.*rand(sum(id),3));
    
end

gt_struct.frwd_model=generate_LFM_library_CPU(gt_struct.centers, psf_ballistic, radius, [1 1 4], vol_movie_size(1:3));
gt_struct.kernel = generate_kernel('ball',ceil(radius*[1 1/4]));

spiketimes = randperm(vol_movie_size(end));
spiketimes = spiketimes(1:dim);

gt_struct.spikes = zeros(num_neurons,vol_movie_size(end));

for ii=1:num_neurons
   SP = spiketimes + round(ceil(log(1000)/Ca_decay)*randn(size(spiketimes)));
   SP = SP(SP>0);
   SP = SP(SP<vol_movie_size(end));
   sample = rand(1,length(SP))>1-3/dim;
   gt_struct.spikes(ii,SP(logical(sample)))=1;
end

gt_struct.Ca_kernel = generate_Ca_kernel(Ca_decay);

gt_struct.timeseries_noisefree = conv2(gt_struct.spikes,gt_struct.Ca_kernel,'same') + 1 + ceil(9*rand(num_neurons,1));

gt_struct.timeseries = gt_struct.timeseries_noisefree + noiselevel * rand(size(gt_struct.timeseries_noisefree));


gt_struct.Volume = zeros(m_size);

for ii=1:num_neurons
    gt_struct.Volume(gt_struct.centers(ii,1),gt_struct.centers(ii,2),gt_struct.centers(ii,3))=1;
end

gt_struct.Volume = gather(convn(gpuArray(gt_struct.Volume),gpuArray(gt_struct.kernel),'same'));

gt_struct.movie = gt_struct.frwd_model'*gt_struct.timeseries;

gt_struct.movie = gt_struct.movie/max(gt_struct.movie(:));

gt_struct.movie = gt_struct.movie.*(exp(-bleach_decay*[0:size(gt_struct.movie,2)-1])+300);

end