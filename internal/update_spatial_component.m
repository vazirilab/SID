function [forward_model]=update_spatial_component(timeseries, sensor_movie, template, opts)
% UPDATE_SPATIAL_COMPONENT performs an update of the spatial components, by
% splitting the problem in sub-problems defined by 'template' and solving
% for each of those sub-problems a non-negative least squares problem.
%
% Input:
% timeseries...         Array of timeseries
% sensor_movie...       LFM-movie
% template...           binary array assigning each neuron its spatial
%                       extent.
% struct opts:
% opts.bg_sub...        perform background subtraction, treat the problem
%                       so that the last component of timeseries is treated 
%                       as the background.
% opts.display...       boolean, if true print status information into the
%                       console.
% opts.lambda...        lagrange multiplier for L1-regularizer.
%
% Output:
% forward_model...      updated forward_model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(opts,'bg_sub')
    opts.bg_sub=1;
end

if ~isfield(opts,'display')
    opts.display=0;
end

[~, order]=sort(sum(template,2));

I=[];
J=[];
S=[];

outside = ~max(template,[],1);

for neur=1:size(template,1)
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
        Y=sensor_movie(space,:);
        
        opts.Accy=0;
        A=timeseries(involved_neurons,:)';
        F=zeros(length(involved_neurons),size(space,2));
        for k_=1:length(space)
            idx=find(squeeze(temp(:,k_)));
            y=squeeze(Y(k_,:))';
            nrm=norm(y(:));
            if nrm>0
                x_ = reg_nnls(A(:,idx),y/nrm,opts);
                %                 x_ = nnls(A(:,idx),y/nrm,opts);
                F(idx,k_) = x_*nrm;
            else
                F(idx,k_) = 0;
            end
        end       
        if size(involved_neurons,2)>=1
            template(:,space)=0;
            [iI, iJ, iS]=find(F);
            iJ=space(iJ);
            iI=involved_neurons(iI);
            iS=reshape(iS,length(iS),1);
            I=[I reshape(iI,1,[])];
            J=[J reshape(iJ,1,[])];
            S=[S' iS']';
        end
    end
    if opts.display
        disp(neuron);
    end
end

if isfield(opts,'gpu')
    if opts.gpu
        gpuDevice([]);
    end
end
S=double(S);
forward_model=sparse(I,J,S,size(timeseries,1),size(sensor_movie,1));

if opts.bg_sub&&~isempty(outside)
    Y=sensor_movie(logical(outside),:);
    forward_model(end,logical(outside))=Y*timeseries(end,:)';
end

end