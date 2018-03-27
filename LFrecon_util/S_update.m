function [S,T]=S_update(Y,S,T,opts)
T(isnan(T))=0;


Q_T = T*T';
q_S = Y*T';

if opts.lamb_orth_L1
    Q_T = Q_T + opts.lamb_orth_L1*(opts.hilf);
end

if opts.lamb_spat_TV
    lb_spat_TV =@(X) opts.lamb_spat_TV*reshape(convn(reshape(X',opts.rank,opts.size(1),...
        opts.size(2)),laplace,'same'),opts.rank,[])';
else
    lb_spat_TV =@(X) 0;
end

df_S = -q_S + S*Q_T + lb_spat_TV(S) + opts.lamb_spat;

if (opts.lamb_orth_L1 + opts.lamb_orth_L2)
    if opts.lamb_orth_L2
        v = df_S + opts.lamb_orth_L2*(S*(opts.hilf*(S'*S)));
    else
        v = df_S;
    end
    z = sum(S.*v,2);
    v = v - S.*z;
    passive_S = max(S>0,v<0);
    v = passive_S.*v;
else
    passive_S = max(S>0,df_S<0);
    v = passive_S.*df_S;
end
    
if opts.pointwise
    alpha_S = sum(v.*df_S,2)./sum(v.*(v*Q_T + lb_spat_TV(v)),2);
else
    alpha_S = (v(:)'*df_S(:))/sum(sum(v.*(v*Q_T + lb_spat_TV(v)),1),2);
    if (alpha_S<0)&&(opts.lamb_orth_L2~=0)
        alpha_S = 1e-6;
    end
end

alpha_S(isnan(alpha_S))=0;
alpha_S(isinf(alpha_S))=0;

S = S - alpha_S.*v;

S(S<0)=0;

if opts.lamb_orth_L1 + opts.lamb_orth_L2
        platz = sqrt(sum(S.^2,1));
        T = platz'.*T;
        S = S./platz;
end

if opts.diagnostic
    figure(1);
    clf('reset')
    for i=1:min(10,opts.rank)
        subplot(3,5,i);
        title(['spat. comp: ' num2str(i)]);
        imagesc(reshape(S(:,i),size(opts.active)));
        axis image;
    end
    legend('boxoff');
    drawnow expose
end

end