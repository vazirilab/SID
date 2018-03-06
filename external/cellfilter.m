function filtered=cellfilter(stack,sigma,znonuniformity,id)

a=1.6;

filter1o=fspecial3_mod('gaussian',[floor(sigma/2)*2+1,floor(sigma/2)*2+1,max(3,floor(sigma/znonuniformity/2)*2+1)],[sigma,sigma,sigma/znonuniformity]);
filter2o=fspecial3_mod('gaussian',[floor(sigma*a/2)*2+1,floor(sigma*a/2)*2+1,max(3,floor(sigma*a/znonuniformity/2)*2+1)],[sigma*a,sigma*a,sigma*a/znonuniformity]);

if size(filter1o,1)>size(filter2o,1)
    filter2o=padarray(filter2o,[size(filter1o,1)-size(filter2o,1),0,0]/2);
elseif size(filter1o,1)<size(filter2o,1)
    filter1o=padarray(filter1o,[size(filter2o,1)-size(filter1o,1),0,0]/2);
end

if size(filter1o,2)>size(filter2o,2)
    filter2o=padarray(filter2o,[0, size(filter1o,2)-size(filter2o,2),0]/2);
elseif size(filter1o,2)<size(filter2o,2)
    filter1o=padarray(filter1o,[0, size(filter2o,2)-size(filter1o,2),0]/2);
end

if size(filter1o,3)>size(filter2o,3)
    filter2o=padarray(filter2o,[0, 0, size(filter1o,3)-size(filter2o,3)]/2);
elseif size(filter1o,3)<size(filter2o,3)
    filter1o=padarray(filter1o,[0, 0, size(filter2o,3)-size(filter1o,3)]/2);
end

if id>0
    filtered=gather(convn(gpuArray(stack),gpuArray(filter1o-filter2o),'same'));
else
    filtered=convn(stack,filter1o-filter2o,'same');
end

end

