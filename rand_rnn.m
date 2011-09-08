n = 20;
t = 10;
m = 10;
k = 10;

W_hh = 0.1 * randn(n) .* (rand(n) > 0.9);
W_hx = randn(n,m) .* (rand(n,m) > 0.9);
W_yh = 0.1 * randn(k,n) .* (rand(k,n) > 0.9);

b_h = 0.1 * randn(n,1);
b_y = 0.1 * randn(k,1);

[y,h,t,s] = rnn( randn(m,t), randn(n,1), W_hh, W_hx, W_yh, b_h, b_y, @tanh, @(x) exp(x)/sum(exp(x)) );