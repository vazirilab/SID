function [S,T]=T_update(Y,T,S,opts)
S(isnan(S))=0;
T(isnan(T))=0;
% line = ~logical(sum(S,1)); 
% if max(line)
%     [S,T]=S_update(Y,S,T,opts);
%     disp('zero line detected');
% end

if opts.lamb_corr>0
    for u=1:size(T,1)
        platz = norm(T(u,:));
        T(u,:) = T(u,:)/platz;
        S(:,u) = S(:,u)*platz;
    end
end

Q_S = S(opts.active,:)'*S(opts.active,:);
q_T = S(opts.active,:)'*Y(opts.active,:);

if opts.use_std
    Q = @(x) Q_S*x - (Q_S*sum(x,2))/size(T,2);
    q_T = q_T - S(opts.active,:)'*opts.Y(opts.active,1)/size(T,2);
else
    Q = @(x) Q_S*x;
end

if opts.lamb_corr
    zsc = zscore(T);
    N =size(T,2);
    d = std(T,[],2);
    hilf = ((zsc*zsc')*zsc - zsc);
    hilf = diag(1./d)*(hilf - (diag(sum(hilf,2))*ones(size(T))/N + (diag(diag(hilf*zsc'))*zsc)/N));
    df_T = -q_T + Q(T) + opts.lamb_temp + opts.lamb_corr*hilf;
else
    df_T = -q_T + Q(T) + opts.lamb_temp;
end

if opts.lamb_temp_TV
    hilf=zeros(size(T)+[0,2]);
    hilf(:,2:end-1)=T;
    hilf(:,1)=T(:,2);
    hilf(:,end)=T(:,end-1);
    hilf=conv2(hilf,[-1 2 -1],'full');
    hilf= hilf(:,3:end-2);
   df_T = df_T + opts.lamb_temp_TV*hilf;
end

passive_T = max(T>0,df_T<0);

df_T_ = passive_T.*df_T;

if opts.pointwise
    alpha_T = sum(df_T_.^2,1)./sum(df_T_.*(Q(df_T_)),1);
else
    alpha_T = sum(df_T_(:).^2)/sum(sum(df_T_.*(Q(df_T_)),1),2);
end

alpha_T(isnan(alpha_T))=0;
alpha_T(isinf(alpha_T))=0;

if ~max(isnan(alpha_T(:)))
    T = T - df_T_.*alpha_T;
end
T(T<0)=0;


if opts.diagnostic
ts = zscore(T(1:10,:), 0, 2);
y_shift = 4;
clip = true;

    sel = 1:size(ts,1);

nixs = 1:size(ts,1);
sel_nixs = nixs(sel);

figure(2);
subplot(121);
hold off
for n_ix = 1:floor(numel(sel_nixs)/2)
    ax = gca();
    ax.ColorOrderIndex = 1;
    loop_ts = ts(sel_nixs(n_ix),:);
    if clip
        loop_ts(loop_ts > 3*y_shift) = y_shift;
        loop_ts(loop_ts < -3*y_shift) = -y_shift;
    end
    t = (0:size(ts,2)-1);
    plot(t, squeeze(loop_ts) + y_shift*(n_ix-1));
    hold on
end
xlabel('Frame');
xlim([min(t) max(t)]);
hold off;
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
legend('boxoff');

subplot(122);
hold off
for n_ix = ceil(numel(sel_nixs)/2):numel(sel_nixs)
    ax = gca();
    ax.ColorOrderIndex = 1;
    loop_ts = ts(sel_nixs(n_ix),:);
    if clip
        loop_ts(loop_ts > y_shift) = y_shift;
        loop_ts(loop_ts < -y_shift) = -y_shift;
    end
    t = (0:size(ts,2)-1);
    plot(t, squeeze(loop_ts) + y_shift*(n_ix-1));
    hold on;
end
xlabel('Frame');
xlim([min(t) max(t)]);
hold off;
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
legend('boxoff');
drawnow expose
end


end