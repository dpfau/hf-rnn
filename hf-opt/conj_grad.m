function params1 = conj_grad( params0, grad, hess_v, p0 )
% Linear conjugate gradient ascent, for the inner loop of HF optimization
% Minimizes 1/2*x'*A*x - b'*x
% params0 - cell array with initial position of x
% grad - cell array with b
% hess_v - function handle that returns A*x given x
% p0 - cell array with initial search direction
% params1 - final value of x

params1 = params0;
r0 = cellfun( @minus, grad, hess_v( params0 ), 'UniformOutput', 0 );
if nargin < 4
    p = r0;
else
    p = p0;
end

i = 1;
objs = zeros(10,1); % tracks the last several values of 1/2x'Ax - b'x
eps = 0.00005; % cutoff for rate of change of objective
while 1
    Ap = hess_v( p );
    a = celldot( r0, r0 ) / celldot( p, Ap );
    params1 = cellfun( @(x,y) x + a*y, params1, p, 'UniformOutput', 0 );
    
    % check if cutoff condition is met
    obj = 1/2 * celldot( params1, hess_v( params1 ) ) - celldot( params1, grad );
    fprintf( 'Obj = %d, Res = %d\n', obj, celldot(r0,r0) );
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
    
    r1 = cellfun( @(x,y) x - a*y, r0, Ap, 'UniformOutput', 0 );
    b = celldot( r1, r1 ) / celldot( r0, r0 );
    p = cellfun( @(x,y) x + b*y, r1, p, 'UniformOutput', 0 );
    r0 = r1;
    i = i + 1;
end