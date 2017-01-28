function template=generate_template(forward_model,Nnum,thres,dim)

disp('Generate template');
II=[];JJ=[];
tic
for neuron=1:size(forward_model,1)
    img=reshape(forward_model(neuron,:),dim);
    img_=zeros(dim/Nnum);
    for k=1:dim(1)/Nnum
        for j=1:dim(2)/Nnum
            img_(k,j)=mean(mean(img((k-1)*Nnum+1:k*Nnum,(j-1)*Nnum+1:j*Nnum)));
        end
    end
    img_=img_/max(img_(:));
    img_(img_<thres)=0;
    [I_,J_,~]=find(img_);
    I=[];J=[];
    for k=1:length(I_)
        s=(I_(k)-1)*Nnum+1:I_(k)*Nnum;
        for l=1:Nnum
            I=[I' (ones(Nnum,1)*s(l))']';
            J=[J' ((J_(k)-1)*Nnum+1:J_(k)*Nnum)]';
        end
    end
    II=[II' (ones(size(I))*neuron)']';
    JJ=[JJ' sub2ind(size(img),I,J)']';
    %     disp(neuron);
end
toc
template=sparse(II,JJ,ones(size(II)),size(forward_model,1),prod(dim));
toc
disp([num2str(neuron) ' templates generated']);

end