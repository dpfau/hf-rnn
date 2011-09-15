function t = diag_slice(x,dim)
% returns a rank-3 tensor where the matrix x is assigned to the indices 
% *other than* dim which are the same, that is, a diagonal slice parallel
% to dim

n = length(x);
t = zeros(n,n,n);
if dim == 1
    t( ones(n,1)*(1:n) + (0:n*(n+1):n^3)'*ones(1,n) ) = x;    
elseif dim == 2
    t( ones(n,1)*(1:n:n^2) + (0:n^2+1:n^3)'*ones(1,n) ) = x;
elseif dim == 3
    t( ones(n,1)*(1:n+1:n^2) + (0:n^2:n^2*(n-1))'*ones(1,n) ) = x;
end