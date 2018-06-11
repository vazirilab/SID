function kernel=generate_Ca_kernel(decay)

kernel = zeros(1,2*ceil(log(1000)/decay)+1);

kernel(ceil(log(1000)/decay)+1:end)=exp(-decay*(0:ceil(log(1000)/decay)));

kernel = kernel/sum(kernel);

end