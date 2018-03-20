function [S,T]=S_update(Y,S,T,opts)
T(isnan(T))=0;

% line = ~logical(sum(T,2));
% if max(line)
%     T(line,:)=conv2(rand(sum(line),size(T,2)),ones(1,32),'same');
%     disp('zero line detected');
% end


Q_T = T*T';
q_S = Y*T';
n=size(Q_T);


if opts.lamb_orth
    hilf = (S*(opts.hilf.*(S'*S)));
    df_S = -q_S + S*Q_T + opts.lamb_spat + opts.lamb_orth*hilf;
else
    df_S = -q_S + S*Q_T + opts.lamb_spat;
end

if opts.lamb_spat_TV
    df_S = df_S + opts.lamb_spat_TV*reshape(convn(reshape(S',n,opts.size(1),opts.size(2)),laplace,'same'),n,[])';
end

if opts.lamb_orth
    z = sum(S.*df_S,2);
    df_S = df_S - S.*z;
end

passive_S = max(S>0,df_S<0);

df_S_ = passive_S.*df_S;

if opts.pointwise
    alpha_S = sum(df_S_.^2,2)./sum(df_S_.*(df_S_*Q_T),2);
else
    alpha_S = sum(df_S_(:).^2)/sum(sum(df_S_.*(df_S_*Q_T),1),2);
end

alpha_S(isnan(alpha_S))=0;
alpha_S(isinf(alpha_S))=0;

if ~max(isnan(alpha_S(:)))

    S = S - alpha_S.*df_S_;
end

S(S<0)=0;

if opts.lamb_orth
    for u=1:size(T,1)
        platz = norm(S(:,u));
        T(u,:) = T(u,:)*platz;
        S(:,u) = S(:,u)/platz;
    end
end

if opts.diagnostic
for ii=1:min(10,n)
figure(ii); imagesc(reshape(S(:,ii),size(opts.active)));
axis image;
end
drawnow expose
end

end