function x = hf_opt( x0, f, grad, hess_v, lambda, mu, maxiters )
% Hessian-free optimization of the function f starting at x0
% x0 - initial value of optimization (for example, current weights of a NN)
% f  - function we are trying to minimize, for example @(x) L(rnn(a,x),b)
%      where L is loss function of RNN, rnn(a,x) gives estimate of b with
%      input a and weights x, and b is true output we are trying to match
% grad(x) - the gradient of f at x
% hess_v(x,v,lm) - the Hessian at x multiplied by v with damping parameter
%                  lm (or approximation, as the case may be)
% lambda - the damping parameter, which we update heuristically
% maxiters - the number of iterations we run the optimization for
%
% David Pfau, 2011

z = cellfun( @(x) zeros(size(x)), x0, 'UniformOutput', 0 );
x1 = x0;
p = cellfun( @minus, grad( x0 ), hess_v( x0, x0, lambda*mu ), 'UniformOutput', 0 ); % residual
for i = 1:maxiters
    b = cellfun( @(x) -x, grad( x1 ), 'UniformOutput', 0 );
    A = @(v) cellfun( @(x,y) x + lambda * y, hess_v( x1, v, lambda*mu ), v, 'UniformOutput', 0 ); % Hessian-vector multiplication with uniform damping
    [dx p q] = conj_grad( z, b, A, p );
    x = cellfun( @(x,y) x + y, x1, dx, 'UniformOutput', 0 );
    
    rho = ( f( x ) - f( x1 ) ) / q;
    fprintf('f(x) = %d, rho = %d\n', f(x), rho );
    if rho > 3/4
        lambda = lambda * 2/3;
    elseif rho < 1/4
        lambda = lambda * 3/2;
    end
    x1 = x;
end

function [x p obj] = conj_grad( x0, grad, hess_v, p0, mode )
% Linear conjugate gradient ascent, for the inner loop of HF optimization
% Minimizes 1/2*x'*A*x - b'*x
% params0 - cell array with initial position of x
% grad - cell array with b
% hess_v - function handle that returns A*x given x
% p0 - cell array with initial search direction
% params1 - final value of x

x = x0;
r = cellfun( @minus, grad, hess_v( x ), 'UniformOutput', 0 ); % residual
p = p0;

i = 1;
objs = zeros(10,1);
eps = 0.00005; % cutoff for rate of change of objective
while 1
    % Update estimate
    Ap = hess_v( p );
    pAp = celldot( p, Ap );
    a = celldot( r, p ) / pAp;
    x = cellfun( @(x,y) x + a*y, x, p, 'UniformOutput', 0 );
    
    % check if cutoff condition is met
    Ax = hess_v( x );
    r = cellfun( @minus, grad, Ax, 'UniformOutput', 0 ); % Ax - b
    obj = 1/2 * celldot( x, Ax ) - celldot( x, grad ); % 1/2x'Ax - b'x
    if nargin > 4 && strcmp( mode, 'verbose' )
        fprintf( 'Obj = %d, Res = %d\n', obj, celldot( r, r ) );
    end
    if i <= 10
        objs(i) = obj;
    else
        if numel( objs ) < ceil( i/10 )
            objs = [objs; obj]; % extend memory length by 1
        else
            objs(1:end-1) = objs(2:end);
            objs(end) = obj;
        end
        if objs(end) < 0 && (objs(end) - objs(1))/objs(end) < numel(objs)*eps
            break;
        end
    end
    
    % Update search direction
    b = -celldot( r, Ap ) / pAp;
    p = cellfun( @(x,y) x + b*y, r, p, 'UniformOutput', 0 );
    i = i + 1;
end