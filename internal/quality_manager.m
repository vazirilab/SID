function [active,active_, passive,in,new,out,out_,err_,time]=quality_manager(time,err_,active,passive,in,iter,cook,opts)

out=[];


%% generate tests
Xs=[[1:opts.sample]',ones(opts.sample,1)];
k=(Xs'*Xs)\(Xs'*log(err_(:,iter-opts.sample+1:iter)'));

c_1=sqrt(sum((log(err_(:,iter-opts.sample+1:iter,1))'-Xs*k).^2,1))<opts.sample*opts.tol_;       %active
c_1=c_1.*(k(1,:)<0);
c_2=(err_(:,iter)'./(1-exp(k(1,:))));                                                           %active
c_2=c_2./cook;
c_3=(iter>time);                                                                                %passive
c_4=(err_(:,iter)'./cook<1e-30);
%% update
prep=logical(c_2<opts.tol);
out_=[active(logical(prep.*c_1)) passive(logical(c_3)) active(c_4)];
passive=[passive(~logical(c_3)) active(logical(c_1.*(1-prep).*(1-c_4)))];
active_=logical((1-c_1).*(1-c_4));
active=active(active_);

if max(logical(c_1.*(1-prep).*(1-c_4)))==1
    time=[time(logical(1-c_3)) iter+(log(opts.tol)-log(c_2(logical(c_1.*(1-prep).*(1-c_4)))))./k(1,logical(c_1.*(1-prep).*(1-c_4)))];
else
    time=time(logical(1-c_3));
end

new=sort([active passive]);
out=in(out_);
in=in(new);

rev=ones(1,max(new));
for i=1:length(new)
    rev(new(i))=i;
end

active=rev(active);
passive=rev(passive);

end

