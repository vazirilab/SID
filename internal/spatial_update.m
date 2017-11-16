function [forward_model,template,centers, res]=spatial_update(psf_ballistic,forward_model,timeseries, centers, sensor_movie, template, opts)

if ~isfield(opts,'bg_sub')
    opts.bg_sub=1;
end

if ~isfield(opts,'display')
    opts.display=0;
end

template = full(template)>0;

Hs =(size(psf_ballistic.H)-1)/2;
[X1,Y1]=meshgrid(1:opts.vol_size(1),1:opts.vol_size(2));

[~, order]=sort(sum(template,2));

res = zeros(size(timeseries,1),27);
res(:,14)=1;

for neur=1:size(template,1)
    tic
    neuron=order(neur);
    space=find(template(neuron,:));
    if ~isempty(space)
        involved_neurons=find(max(template(:,space),[],2));
        if opts.bg_sub==0
            temp=template(involved_neurons,space);
            if size(template,1)~=size(timeseries,1)
                disp('Error: template and timeseries have the wrong dimensions');
                return
            end
        else
            temp=zeros(length(involved_neurons)+1,length(space));
            temp(1:length(involved_neurons),:)=template(involved_neurons,space);
            temp(length(involved_neurons)+1,:)=ones(1,length(space));
            involved_neurons=[involved_neurons' size(timeseries,1)];
            if size(template,1)==size(timeseries,1)
                disp('Error: template and timeseries have the wrong dimensions');
                return
            end
        end
        involved_neurons = involved_neurons(involved_neurons~=neuron);
        Y=sensor_movie(space,:);
        
        T=timeseries(involved_neurons,:)';
        
        cent_=centers(neuron,:);
        for xx=-1:1
            for yy=-1:1
                for zz=-1:1
                    cent(sub2ind([3 3 3],xx+2,yy+2,zz+2),:) = [cent_(1) + xx,cent_(2) + yy,cent_(3) + zz];
                end
            end
        end
        cent((cent<1))=nan;
        cent((cent(:,1)>opts.vol_size(1)),1)=nan;
        cent((cent(:,2)>opts.vol_size(2)),2)=nan;
        cent((cent(:,3)>opts.vol_size(3)),3)=nan;
        id2=~isnan(sum(cent,2));
        cent=cent(id2,:);
        
        psf = project_forward_pixel(cent, opts.vol_size,psf_ballistic.Nnum, psf_ballistic.H);
        
        Y = Y - forward_model(involved_neurons,space)'*T';
        Y = max(Y*timeseries(neuron,:)'/norm(timeseries(neuron,:)),0);
        q = psf(:,space)*Y;
        psi = psf(:,space)*psf(:,space)';
        R=q;
        for k=1:opts.max_iter
            R = R.*(q./(psi*R));
            R(isnan(R))=0;
        end
        
        res(neuron,id2)=R'/norm(timeseries(neuron,:));
        [~,n_] = max(res(neuron,:));
        centers(neuron,:) = cent(n_,:);
        forward_model(neuron,:) = R'*psf;

        cam = sqrt((X1-centers(neuron,1)).^2 + (Y1-centers(neuron,2)).^2)<...
            ((Hs(1)-psf_ballistic.Nnum*1.5)*abs(opts.native_focal_plane-centers...
            (neuron,3))/max(1+2*Hs(5)-opts.native_focal_plane,opts.native_focal_plane) + psf_ballistic.Nnum*1.5);
        
        template(neuron,:) = cam(:);
        
    end
    if strcmp(opts.display,'on')
        disp(neur);
        toc
    end
end

if opts.bg_sub
Y = sensor_movie - forward_model(1:end-1,:)'*timeseries(1:end-1,:);
forward_model(end,:) = max(0,Y*timeseries(end,:)'/norm(timeseries(end,:)));
end

if isfield(opts,'gpu')
    if strcmp(opts.gpu,'on')
        gpuDevice([]);
    end
end

end

