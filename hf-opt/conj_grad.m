function x = conj_grad( x0, grad, hess_v, p0 )
% Linear conjugate gradient ascent, for the inner loop of HF optimization
% Minimizes 1/2*x'*A*x - b'*x
% params0 - cell array with initial position of x
% grad - cell array with b
% hess_v - function handle that returns A*x given x
% p0 - cell array with initial search direction
% params1 - final value of x

x = x0;
r = cellfun( @minus, grad, hess_v( x ), 'UniformOutput', 0 ); % residual
if nargin < 4
    p = cellfun( @(x) -x, r, 'UniformOutput', 0 );
else
    p = p0;
end

i = 1;
objs = zeros(10,1);
eps = 0.00005; % cutoff for rate of change of objective
while 1
    Ap = hess_v( p );
    pAp = celldot( p, Ap );
    a = celldot( r, p ) / pAp;
    x = cellfun( @(x,y) x + a*y, x, p, 'UniformOutput', 0 );
    
    % check if cutoff condition is met
    Ax = hess_v( x );
    r = cellfun( @minus, grad, Ax, 'UniformOutput', 0 ); % Ax - b
    obj = 1/2 * celldot( x, Ax ) - celldot( x, grad ); % 1/2x'Ax - b'x
    fprintf( 'Obj = %d, Res = %d\n', obj, celldot( r, r ) );
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
    b = celldot( r, Ap ) / pAp;
    p = cellfun( @(x,y) -x + b*y, r, p, 'UniformOutput', 0 );
    i = i + 1;
end