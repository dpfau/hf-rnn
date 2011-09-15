function t = diag_tensor(x,k)
% generates a rank-k tensor that has the vector x along the diagonal

n = length(x);
t = zeros(repmat(n,1,k));
t(1:(n^k - 1)/(n - 1):n^k) = x;