function [forward_model] = update_spatial_component(timeseries, sensor_movie, template, opts)
%% UPDATE_SPATIAL_COMPONENT  Perform an update of the spatial components
% by splitting the problem in sub-problems defined by 'template' and solving
% for each of those sub-problems a non-negative least squares problem.
%
% Input:
% timeseries         Array of timeseries
% sensor_movie       LFM-movie
% template           binary array assigning each neuron its spatial
%                       extent.
% struct opts:
% opts.bg_sub        perform background subtraction, treat the problem
%                       so that the last component of timeseries is treated 
%                       as the background.
% opts.display       boolean, if true print status information into the
%                       console.
% opts.lambda        lagrange multiplier for L1-regularizer.
%
% Output:
% forward_model      updated forward_model

%% Set default values for parameters not set by user
if ~isfield(opts,'bg_sub')
    opts.bg_sub=1;
end

if ~isfield(opts,'display')
    opts.display=0;
end

%% Order neurons according to the size of their template
% this is done so that sub-problems tend to be smaller.
[~, order]=sort(sum(template,2));

%% Initialize components for the sparse matrix generation
I=[];
J=[];
S=[];

%% Determine which part of the image is not covered by the template
% (and therefore needs to be updated separately if opts.bg_sub is true)
outside = ~max(template,[],1);

%% Loop over neurons
for neur=1:size(template,1)
    neuron=order(neur);

    %% Get all indices of pixels where the spatial component can be non-zero according to ‘template’
    space=find(template(neuron,:));

    %% If any such pixels remain proceed to the next line
    if ~isempty(space)
        %% Find all neurons that are non-zero
        involved_neurons=find(max(template(:,space),[],2));

        %% Check if the dimensions of timeseries and template have the right number of components
        % (according to whether opts.bg_sub is true, or not). 
        % Generate the sub array of template corresponding to the indices found above. 
        % If opts.bg_sub is true add a ones() line at the end of the array, to incorporate the background
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

        %% Get the sub-movie corresponding to the indices found above
        Y=sensor_movie(space,:);

        %% Generate the matrix A for the objective function of the sub-problem and F to store the solution of the sub-problem
        A=timeseries(involved_neurons,:)';
        F=zeros(length(involved_neurons),size(space,2));

        %% Loop over each pixel corresponding to indices found above. Solve for each pixel separately
        opts.Accy=0;
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

        %% Add the current results to the components I, J and S for the sparse matrix generation
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

%% Release GPUs
if isfield(opts,'gpu')
    if opts.gpu
        gpuDevice([]);
    end
end

%% Generate sparse matrix
S=double(S);
forward_model=sparse(I,J,S,size(timeseries,1),size(sensor_movie,1));

%% Compute the values for the spatial background outside of the area covered by template
if opts.bg_sub&&~isempty(outside)
    Y=sensor_movie(logical(outside),:);
    forward_model(end,logical(outside))=Y*timeseries(end,:)';
end

end
