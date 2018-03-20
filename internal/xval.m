function options=xval(Y,n,opts)
% XVAL: algorithm performs cross-validation for the parameter
% opts.lamb_orth for the problem defined by Y,n and opts.
%
% Input:
% opts...           options to be modified by xval
% Y...              movie
% n...              rank of the nnmf
% struct opts:
% opts.im_size      size of the std_image of Y
% opts.max_iter     maximal number of iterations of the nnmf inside xval
% opts.num_part     number of partitions in which the data is decomposed
% param             paramter range of opts.lamb_orth that needs to be
%                   scanned by xval
%
% Output:
% options...           output options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(opts.xval,'im_size')
    opts.xval.im_size = 100;
end

if ~isfield(opts.xval,'max_iter')
    opts.xval.max_iter=100;
end

if ~isfield(opts.xval,'num_part')
    opts.xval.num_part=5;
end

if ~isfield(opts.xval,'param')
    opts.xval.param = opts.lamb_orth/size(Y,2)*exp(-2*[0:4]);
end

map = conv2(opts.xval.std_image.^2,ones(opts.xval.im_size+1),'valid');
[~,m] = max(map(:));
[c_x,c_y] = ind2sub(size(map),m);
[cx,cy] = meshgrid(c_y:c_y+opts.xval.im_size,c_x:c_x+opts.xval.im_size);
ind = sub2ind(size(opts.xval.std_image),cx(:),cy(:));
Y_p = Y(ind,:);
option=opts;
option.diagnostic = false;
option.max_iter=2000;
option.active = opts.active(c_y:c_y+opts.xval.im_size,c_x:c_x+opts.xval.im_size);
[t,s] = initialize_nnmf(Y_p,n,option);
if opts.lamb_orth
    for u=1:size(t,1)
        platz = norm(s(:,u));
        t(u,:) = t(u,:)*platz;
        s(:,u) = s(:,u)/platz;
    end
end
[N,~] = discretize([1:size(Y_p,2)],opts.xval.num_part);

for j=1:length(opts.xval.param)
    option.lamb_orth=opts.xval.param(j);
    for k=1:opts.xval.num_part
        yy=Y_p(:,k==N); y=Y_p(:,k~=N); T=t(:,k~=N); S=s;
        option.lamb_orth*size(y,2);
        for iter=1:opts.xval.max_iter
            [S,T]=S_update(y,S,T,option);
            [S,T]=T_update(y,T,S,option);
        end
        for u=1:size(T,1)
            platz = norm(S(:,u));
            T(u,:) = T(u,:)*platz;
            S(:,u) = S(:,u)/platz;
        end
        T = LS_nnls(S,yy,opts);
        E(k) = norm(reshape(S*T-yy,1,[]))/size(yy,2);
        E2(k)= norm(reshape(S'*S-eye(size(S,2)),1,[]));
    end
    disp(j);
    E_(j) = mean(E);
    disp(E_(j));
    E2_(j) = mean(E2);
end
[~,n_] = min(E_);
opts.lamb_orth=opts.lamb_orth*exp(-2*(n_-1));
options = opts;
figure;plot(E_)
figure;plot(E2_)
disp(opts.lamb_orth);
end