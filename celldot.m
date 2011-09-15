function x = celldot( a, b )
% Treats a cell array of arrays as one long vector, and takes the dot
% product between two of them

lin = @(a) cellfun( @(x) x(:), a, 'UniformOutput', 0 );
x = sum( cellfun( @dot, lin(a), lin(b) ) );