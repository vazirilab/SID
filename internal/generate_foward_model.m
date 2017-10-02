function forward_model=generate_foward_model(centers,psf_ballistic,r,rr,m,kernel)
tic
dim=m(1:2);
Hsize=size(psf_ballistic.H);
disp('Initiate forward_model');
forward_model_indices=cell(1,size(centers,1));
forward_model_values=forward_model_indices;
N=0;

BWW=[];

W=zeros(2*r+1,2*r+1,2*rr+1);

if nargin==6
    W = kernel;
    r = (size(W,1)-1)/2;
    rr = (size(W,3)-1)/2;
else
    for ii=1:2*r+1
        for jj=1:2*r+1
            for kk=1:2*rr+1
                if  ((ii-(r+1))^2/r^2+(jj-(r+1))^2/r^2+(kk-(rr+1))^2/rr^2)<=1
                    W(ii,jj,kk)=1;
                end
            end
        end
    end
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
    if mod(k, 20) == 1
        fprintf([num2str(k) ' ']);
    end
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
end
forward_model=sparse(I,J,S,size(centers,1),prod(dim));
fprintf('\n');
toc;
disp([num2str(size(centers,1)) ' NSFs generated']);
end