function [recon_NSF, x,y,z]=fast_NSF_recon(forward_model,centers,psf_ballistic, size_std,options)

%% NSF reconstruction

disp('Reconstructing NSFs');
poolobj = gcp('nocreate');
delete(poolobj);

if nargin<5
    options=[];
    options.num_Workers=8;
    options.p=2;
    options.maxIter=3;
    options.mode='basic';
    options.gpu_ids=[1 2 4 5];
end

if ~isfield(options,'num_Workers')
    options.num_Workers=6;
end
if ~isfield(options,'p')
    options.p=2;
end
if ~isfield(options,'maxIter')
    options.maxIter=6;
end
if ~isfield(options,'mode')
    options.mode='TV';
end
if ~isfield(options,'lambda')
    options.lambda=[ 0, 0, 10];
end
if ~isfield(options,'lambda_')
    options.lambda_=0.1;
end
if ~isfield(options,'form')
    options.form='gaussian';
end
if ~isfield(options,'rad')
    options.rad=[3,1];
end
if ~isfield(options,'gpu_ids')
    options.gpu_ids=[1,2,4];
end

gimp=options.gpu_ids;
parpool(options.num_Workers);


Nnum=psf_ballistic.Nnum;

for kk=1:options.num_Workers:size(centers,1)
    img=cell(options.num_Workers,1);    
    for worker=1:min(options.num_Workers,size(centers,1)-(kk-1))
        k=kk+worker-1;
        img_=reshape(forward_model(k,:),size_std);
        
        
        xx{1}{worker} = max(1,centers(k,1)-size(psf_ballistic.H,1));
        xx{2}{worker} = min(size_std(1),centers(k,1)+size(psf_ballistic.H,1));
        yy{1}{worker} = max(1,centers(k,2)-size(psf_ballistic.H,1));
        yy{2}{worker} = min(size_std(2),centers(k,2)+size(psf_ballistic.H,1));
        zz{1}{worker} = max(1,round(centers(k,3))-5);
        zz{2}{worker} = min(size(psf_ballistic.H,5),round(centers(k,3))+5);
        
        xx{1}{worker}=floor(xx{1}{worker}/Nnum)*Nnum+1;
        yy{1}{worker}=floor(yy{1}{worker}/Nnum)*Nnum+1;
        xx{2}{worker}=ceil(xx{2}{worker}/Nnum)*Nnum;
        yy{2}{worker}=ceil(yy{2}{worker}/Nnum)*Nnum;
        img_=img_(xx{1}{worker}:xx{2}{worker},yy{1}{worker}:yy{2}{worker});
        img_ = img_-mean(img_(logical(img_)));
        img_(img_<0)=0;
        
        psf{worker}.H=psf_ballistic.H(:,:,:,:,zz{1}{worker}:zz{2}{worker});
        psf{worker}.CAindex=psf_ballistic.CAindex(zz{1}{worker}:zz{2}{worker});
        psf{worker}.Nnum=Nnum;
        
        
        img{worker}=full(img_)/max(img_(:));
    end
    opts=cell(min(options.num_Workers,size(centers,1)-(kk-1)),1);
    recon=cell(min(options.num_Workers,size(centers,1)-(kk-1)),1);
    parfor worker=1:min(options.num_Workers,size(centers,1)-(kk-1))
        
        infile=struct;
        infile.LFmovie=(img{worker});
        opts{worker}=options;
        opts{worker}.gpu_ids=mod((worker-1),length(gimp))+1;
        opts{worker}.gpu_ids=gimp(opts{worker}.gpu_ids);
        
        recon{worker}= reconstruction_new(infile, psf{worker}, opts{worker});
        gpuDevice([]);
    end
    for kp=1:min(options.num_Workers,size(centers,1)-(kk-1))
        recon_NSF{kk+kp-1}=recon{kp};
        x{1}{kk+kp-1}=xx{1}{kp};
        y{1}{kk+kp-1}=yy{1}{kp};
        x{2}{kk+kp-1}=xx{2}{kp};
        y{2}{kk+kp-1}=yy{2}{kp};
        z{1}{kk+kp-1}=zz{1}{kp};
        z{2}{kk+kp-1}=zz{2}{kp};
    end
    disp(kk)
end

end
