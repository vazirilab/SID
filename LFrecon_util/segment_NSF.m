function [forward_model,centers]=segment_NSF(recon_NSF,x,y,z,psf,dim,p)

%% NSF reconstruction

disp('Reconstructing NSFs');
poolobj = gcp('nocreate');
delete(poolobj);

if nargin<5
    options=[];
    options.num_Workers=8;
    options.p=2;
    options.maxIter=3;
    options.mode='TV';
    options.lambda=[ 0, 0, 10];
    options.lambda_=0.1;
    options.form='gaussian';
    options.rad=[3,1];
    options.gpu_ids=[1 2 4 5];
end

if ~isfield(options,'num_Workers')
    options.num_Workers=6;
end

if ~isfield(options,'gpu_ids')
    options.gpu_ids=[1,2,4];
end

%%
for n=1:size(recon_NSF,2)
    conn_comp=bwconncomp(max(recon_NSF{n}-0.1*max(recon_NSF{n}(:)),0));
    
    
end


%%

gimp=options.gpu_ids;
parpool(options.num_Workers);
forward_model=zeros(size(recon_NSF,2),dim(1)*dim(2));

for kk=1:options.num_Workers:size(centers,1)
    Volume=cell(options.num_Workers,1);
    H=cell(options.num_Workers,1);
    frwd=cell(options.num_Workers,1);
    for worker=1:min(options.num_Workers,size(centers,1)-(kk-1))
        
        xx{1}{worker}=x{1}{kk+worker-1};
        xx{2}{worker}=x{2}{kk+worker-1};
        yy{1}{worker}=y{1}{kk+worker-1};
        yy{2}{worker}=y{2}{kk+worker-1};
        zz{1}{worker}=z{1}{kk+worker-1};
        zz{2}{worker}=z{2}{kk+worker-1};
        Volume{worker}=zeros([dim(1:2) size(recon_NSF{kk+worker-1},3)]);       
        Volume{worker}(xx{1}{worker}:xx{2}{worker},yy{1}{worker}:yy{2}{worker},:)...
            =max(recon_NSF{kk+worker-1}-p*max(recon_NSF{kk+worker-1}(:)),0);       
        H{worker}=psf.H(:,:,:,:,zz{1}{worker}:zz{2}{worker});      
    end
    
    parfor worker=1:min(options.num_Workers,size(recon_NSF,2)-(kk-1))
        tic
        gpuDevice(gimp(mod((worker-1),length(gimp))+1));
        frwd{worker}=gather(forwardProjectGPU(gpuArray(H{worker}), gpuArray(Volume{worker})));
        toc
        gpuDevice([]);
    end
    
    for kp=1:min(options.num_Workers,size(centers,1)-(kk-1))
        forward_model(kk+kp-1,:)=frwd{kp};
    end
    disp(kk)
end

end
