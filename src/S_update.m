function [S, T] = S_update(Y, S, T, opts)
%% S_update  Perform a gradient descent with exact line search update for the variables in S

T(isnan(T))=0;
S(isnan(S))=0;

%% Compute components of the gradient of the 2-norm of Y-S*T
Q_T = T*T';
q_S = Y*T';

%%
if opts.use_std
    opts.T = sum(T,2);
    Q_T = Q_T - opts.T * opts.T'/size(T,2);
    q_S = q_S - opts.Y.*opts.T'/size(T,2);
end

%% Modify matrix Q_T to include the contribution of the 1-norm orthogonality regularizer
if opts.lamb_orth_L1
    Q_T = Q_T + opts.lamb_orth_L1*(opts.hilf);
end

%% Generate a function handle that computes the contribution of the spatial Total Variation regularizer
if opts.lamb_spat_TV
    lb_spat_TV =@(X) opts.lamb_spat_TV*reshape(convn(reshape(X',opts.rank,opts.size(1),...
        opts.size(2)),laplace,'same'),opts.rank,[])';
else
    lb_spat_TV =@(X) 0;
end

%% Assemble gradient from its components
df_S = -q_S + S*Q_T + lb_spat_TV(S) + opts.lamb_spat;

%% Generate the direction v in which the update shall be performed
% v is generated from the gradient by projecting along the normalization constraint 
% that is imposed when using an orthogonality regularizer, 
% and by projecting onto the surface of the non-negativity constraint
if (opts.lamb_orth_L1 + opts.lamb_orth_L2) > 0
    if opts.lamb_orth_L2
        % Final assembly of the gradient for the 2-norm orthogonality regularizer 
        % (indirectly, via modification of the direction v)
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

%% Exact line search for the direction v
% In the case of the 2-norm orthogonality regularizer, the exact line search is approximated by exact line search for the residual gradient only. 
% If this results in a value that leads into the opposing direction of the negative gradient, the learning rate is set to a fixed value 1e-6. 
% This is done since the corresponding regularizer leads to a non-quadratic problem.
if opts.pointwise
    alpha_S = sum(v .* df_S, 2) ./ sum(v .* (v * Q_T + lb_spat_TV(v)), 2);
else
    alpha_S = (v(:)' * df_S(:)) / sum(sum(v .* (v * Q_T + lb_spat_TV(v)), 1), 2);
    if opts.lamb_orth_L2 ~= 0
        alpha_S = max(alpha_S, 1e-6);
    end
end
alpha_S(isnan(alpha_S))=0;
alpha_S(isinf(alpha_S))=0;

%% Update S
S = S - alpha_S .* v;

%% Project onto constraints
S(S<0) = 0;

if (opts.lamb_orth_L1 + opts.lamb_orth_L2) > 0
        platz = sqrt(sum(S .^ 2, 1));
        T = platz' .* T;
        S = S ./ platz;
end

%% Output diagnostic info
if opts.diagnostic
    fh = findobj('Type', 'Figure', 'Name', 'S update plot');
    if isempty(fh)
        figure('Name', 'S update plot', 'Position', [10 10 1500 1500]);
    else
        set(0, 'CurrentFigure', fh);
    end
    %clf('reset')
    for i = 1:min(9, opts.rank)
        subplot(3, 3, i);
        imagesc(reshape(S(:,i),size(opts.active)));
        axis image;
        title(['spat. comp: ' num2str(i)]);
    end
    legend('boxoff');
    drawnow expose
end

end
