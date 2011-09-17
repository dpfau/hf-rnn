% Run Hessian-free optimization for a random RNN.  David Pfau, 2011

n = 5;
m = 3;
k = 3;
t = 10;

W_hh = 0.1 * randn(n);
W_hx = randn(n,m);
W_yh = 0.1 * randn(k,n);

b_h = 0.1 * randn(n,1);
b_y = 0.1 * randn(k,1);

x = randn(m,t);
y = rand(k,t);
y = y./(ones(k,1)*sum(y));
h0 = randn(n,1);

params = { h0, W_hh, W_hx, W_yh, b_h, b_y };

g = @tanh;
Jg = @(x) diag(1 - tanh(x).^2);

f = @(params) XH( y, rnn( x, params, g, @SMX ) );
grad = @(params) bptt( x, y, params, g, @SMX, Jg, @dSMX, @dXH );
hess = @(params, v) gauss_newton_v( x, y, params, v, g, @SMX, Jg, @dSMX, @ddSMX, @dXH, @ddXH );

params1 = hf_opt( params, f, grad, hess, 1, 100 );