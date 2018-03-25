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
        disp('help');
    end
end

alpha_S(isnan(alpha_S))=0;
alpha_S(isinf(alpha_S))=0;

S = S - alpha_S.*v;

S(S<0)=0;

if opts.lamb_orth_L1 + opts.lamb_orth_L2
    for u=1:size(T,1)
        platz = norm(S(:,u));
        T(u,:) = T(u,:)*platz;
        S(:,u) = S(:,u)/platz;
    end
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
% 
% 
% % line = ~logical(sum(T,2));
% % if max(line)
% %     T(line,:)=conv2(rand(sum(line),size(T,2)),ones(1,32),'same');
% %     disp('zero line detected');
% % end
% 
% Q_T = T*T';
% q_S = Y*T';
% 
% 
% if opts.lamb_orth_L1
% %     hilf = (S*(opts.hilf.*(S'*S)));
% %     df_S = -q_S + S*Q_T+ opts.lamb_spat;
% %     v = df_S  + opts.lamb_orth_L1*hilf;
% Q_T = Q_T + opts.lamb_orth_L1*(opts.hilf);
% 
% df_S = -q_S + S*Q_T + opts.lamb_spat;
% %     z = sum(S.*v,2);
% %     v = v - S.*z;
% 
% else
%     df_S = -q_S + S*Q_T + opts.lamb_spat;
% end
% 
% if opts.lamb_spat_TV
%     lb_spat_TV =@(X) opts.lamb_spat_TV*reshape(convn(reshape(X',opts.rank,opts.size(1),...
%         opts.size(2)),laplace,'same'),opts.rank,[])';
%     df_S = df_S + lb_spat_TV(S);
% else
%     lb_spat_TV =@(X) 0;
% end
% 
% % if opts.lamb_orth_L1
% % % 	passive_S = max(S>0,v<0);
% % %     v = passive_S.*v;
% % %     alpha_S = (v(:)'*df_S(:))/sum(sum(v.*(v*Q_T),1),2);
% % %     if alpha_S<0
% % %        alpha_S = 1e-12; 
% % %     end
% % else
%     z = sum(S.*df_S,2);
%     df_S = df_S - S.*z;
%     passive_S = max(S>0,df_S<0);
%     v = passive_S.*df_S;
% if opts.pointwise
%     alpha_S = sum(v.^2,2)./sum(v.*(v*Q_T + lb_spat_TV(v)),2);
% else
%     alpha_S = sum(v(:).^2)/sum(sum(v.*(v*Q_T + lb_spat_TV(v)),1),2);
% end
% % end
% 
% alpha_S(isnan(alpha_S))=0;
% alpha_S(isinf(alpha_S))=0;
% 
% if ~max(isnan(alpha_S(:)))
% % if opts.lamb_orth_L1
% %     S = S - alpha_S.*v;    
% % else
%     S = S - alpha_S.*v;
% % end
% end
% 
% S(S<0)=0;
% 
% if opts.lamb_orth_L1
%     for u=1:size(T,1)
%         platz = norm(S(:,u));
%         T(u,:) = T(u,:)*platz;
%         S(:,u) = S(:,u)/platz;
%     end
% end
% 
% if opts.diagnostic
%     figure(1);
%     clf('reset')
%     for i=1:min(10,n)
%         subplot(3,5,i);
%         title(['spat. comp: ' num2str(i)]);
%         imagesc(reshape(S(:,i),size(opts.active)));
%         axis image;
%     end
%     legend('boxoff');
%     drawnow expose
% end
% 
% end