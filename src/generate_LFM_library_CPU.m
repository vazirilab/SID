function forward_model=generate_LFM_library_CPU(centers,psf_ballistic,r,dim,m,kernel)
% GENERATE_LFM_LIBRARY: Algorithm generates a library of LFM-patterns for every neuron location
% found in "centers".
%
% Input:
% centers...            Nx3 Array of 3d coordinates of putative neurons
% psf_ballistic...      point spread function
% r...                  Neuron radius
% dim...                scaling of x, y and z dimension in Volume
% m...                  size of Volume
% kernel...             Neuron kernel, if not set, neuron form will be set
%                       to a ball of radius r.
%
% Output:
% forward_model...      Library of LFM-patterns
%

rr=r/dim(3);

tic
dim=m(1:2);
Hsize=size(psf_ballistic.H);
disp('Initiate forward_model');
forward_model_indices=cell(1,size(centers,1));
forward_model_values=forward_model_indices;
N=0;
BWW=[];

if nargin==6
    W = kernel;
    r = (size(W,1)-1)/2;
    rr = (size(W,3)-1)/2;
else
    [X,Y,Z] = meshgrid(-r:r,-r:r,-rr:rr);
    W = single((X.^2 + Y.^2 + (Z*r/rr).^2)<=r^2);   
end

BW=find(W);
[BWW(:,1), BWW(:,2), BWW(:,3)] = ind2sub([2*r+1,2*r+1,2*rr+1],BW);
for k=1:size(centers,1)
    B = [];
    B_ = [];
    for j=1:size(BWW,1)
        bbb = round(BWW(j,:)-[(r+1)*[1 1] (rr +1)]+centers(k,:));
        bbb_ = round(BWW(j,:));
        if (bbb(1)<=m(1))&&(bbb(1)>0)&&(bbb(2)<=m(2))&&(bbb(2)>0)&&(bbb(3)<=Hsize(5))&&(bbb(3)>0)
            B = [B' bbb']';
            B_ = [B_' bbb_']';
        end
    end
    Q=project_forward_patched(B, dim, psf_ballistic.Nnum, psf_ballistic.H,W,B_);
    Q=Q(:);
    Q=Q/norm(Q);
    forward_model_indices{k}=find(Q);
    forward_model_values{k}=Q(forward_model_indices{k});
    N=N+length(forward_model_values{k});
        disp(k);
end
I=zeros(N,1);
J=I;
S=I;
jj=0;
for k=1:size(forward_model_indices,2)
    J(jj+1:jj+size(forward_model_values{k},1))= forward_model_indices{k};
    I(jj+1:jj+size(forward_model_values{k},1))=k*ones(size(forward_model_values{k}));
    S(jj+1:jj+size(forward_model_values{k},1))=forward_model_values{k};
    jj=jj+size(forward_model_values{k},1);
    %     disp(k)
end
forward_model=sparse(I,J,S,size(centers,1),prod(dim));
toc;
disp([num2str(size(centers,1)) ' NSFs generated']);
end