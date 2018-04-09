function x=reg_nnls(A,d,opts)
% REG_NNLS transforms a regularized nnls problem into a non-negative 
% quadratic problem and uses NQP, to solve the quadratic problem.
%
% Input:
% A...                  matrix
% d...                  inhomogenity
% struct opts:
% opts.use_std...       use standard deviation instead of L2-norm for the
%                       residual
% opts.lamb_L1...       Lagrangian multiplier for L1-regularizer
% opts.lamb_L2...       Lagrangian multiplier for L2-regularizer
% opts.lamb_orth_L1...  Lagrangian multiplier for L1-orthogonality-regularizer
%
% Output:
% x...                  Approximation of the solution of the regularized
%                       nnls problem.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<3
    opts = struct;
end

if ~isfield(opts,'use_std')
    opts.use_std=false;
end
if ~isfield(opts,'lamb_L1')
    opts.lamb_L1=0;
end
if ~isfield(opts,'lamb_L2')
    opts.lamb_L2=0;
end
if ~isfield(opts,'lamb_orth_L1')
    opts.lamb_orth_L1=0;
end
if ~isfield(opts,'bg_sub')
    opts.bg_sub=0;
end

if ~isfield(opts,'hilf')
    opts.hilf = ones(size(A,2))-eye(size(A,2));
    if opts.bg_sub
        opts.hilf(1:end,end) = 0;
        opts.hilf(end,1:end) = 0;
    end
end

Q = A'*A;
q = -A'*d;

if opts.use_std
    opts.T = sum(A,1);
    Q = Q - opts.T'*opts.T/size(A,1);
    q = q + opts.T'.*sum(d,1)'/size(A,1);
end

if opts.lamb_L1
    q = q + opts.lamb_L1;
end

if opts.lamb_L2
    Q = Q + opts.lamb_L2*eye(size(Q,1));
end

if opts.lamb_orth_L1
    Q = Q + opts.lamb_orth_L1*(opts.hilf);
end

x=NQP(Q,q,opts);

end
   
