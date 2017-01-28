function forward_model=generate_foward_model(centers,psf_ballistic,r,rr,m)

dim=m(1:2);
Hsize=size(psf_ballistic.H);
disp('Initiate forward_model');
forward_model_indices=cell(1,size(centers,1));
forward_model_values=forward_model_indices;
N=0;

BWW=[];
W=zeros(2*r,2*r,2*rr);
for ii=1:2*r
    for jj=1:2*r
        for kk=1:2*r
            if  ((ii-((2*r-1)/2+1))^2/r^2+(jj-((2*r-1)/2+1))^2/r^2+(kk-((2*rr-1)/2+1))^2/rr^2)<=1
                W(ii,jj,kk)=1;
            end
        end
    end
end

BW=bwconncomp(W);
[BWW(:,1), BWW(:,2), BWW(:,3)]=ind2sub([2*r,2*r,2*rr],BW.PixelIdxList{1,1});
for k=1:size(centers,1)
    B=[];
    for j=1:size(BWW,1)
        bbb=round(BWW(j,:)-[((2*r-1)/2+1)*[1 1] ((2*rr-1)/2+1)]+centers(k,:));
        if (bbb(1)<=m(1))&&(bbb(1)>0)&&(bbb(2)<=m(2))&&(bbb(2)>0)&&(bbb(3)<=Hsize(5))&&(bbb(3)>0)
            B=[B' bbb']';
        end
    end
    Q=project_forward_patched(B, dim, psf_ballistic.Nnum, psf_ballistic.H);
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