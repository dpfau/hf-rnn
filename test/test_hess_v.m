% Numerically verify the correctness of my implementation of
% Hessian-vector multiplication.  David Pfau, 2011

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
h_0 = randn(n,1);

params = { h_0, W_hh, W_hx, W_yh, b_h, b_y };

g = @tanh;
Jg = @(x) diag(1 - tanh(x).^2);
Hg = @(x) diag_tensor(-2*tanh(x).*(1-tanh(x).^2),3);
e = @(x) exp(x)/sum(exp(x)); % softmax
Je = @(x) diag(exp(x)/sum(exp(x))) - exp(x)*exp(x)'/sum(exp(x))^2;
He = @(x) 2*tprod( exp(x)*exp(x)', [1 2], exp(x), 3 )/sum(exp(x))^3 ...
    + diag_slice( -exp(x)*exp(x)'/sum(exp(x))^2, 1 ) ...
    + diag_slice( -exp(x)*exp(x)'/sum(exp(x))^2, 2 ) ...
    + diag_slice( -exp(x)*exp(x)'/sum(exp(x))^2, 3 ) ...
    + diag_tensor(exp(x),3)/sum(exp(x));

% g = @(x) x;
% Jg = @(x) eye(length(x));
% Hg = @(x) zeros(length(x),length(x),length(x));
% e = @(x) x;
% Je = @(x) eye(length(x));
% He = @(x) zeros(length(x),length(x),length(x));

diff_params = cell(1,6); % the Hessian-vector product computed by difference of gradients
v_params = cell(1,6); % the vector with which we multiply the Hessian
for i = 1:length(diff_params)
    diff_params{i} = zeros(size(params{i}));
    v_params{i} = randn(size(params{i}));
end

eps = 1e-9;

[y0 h0 t0 s0] = rnn( x, params, g, e );
[dparams0 dt0 ds0] = bptt( x, y, params, g, e, Jg, Je, @dXH, y0, h0, t0, s0 );
for i = 1:length(params)
    params{i} = params{i} + eps * v_params{i};
end

[y1 h1 t1 s1] = rnn( x, params, g, e );
[dparams1 dt1 ds1] = bptt( x, y, params, g, e, Jg, Je, @dXH, y1, h1, t1, s1 );
for i = 1:length(params)
    diff_params{i} = ( dparams1{i} - dparams0{i} ) / eps;
    params{i} = params{i} - eps * v_params{i};
end

vars = { 'y', 'h', 't', 's', 'dt', 'ds' };
for i = 1:length(vars)
    var = vars{i};
    eval(['diff_' var ' = ( ' var '1 - ' var '0 ) / eps;'])
end

[Rparams Ry Rh Rt Rs Rdh Rdt Rds] = hess_v( x, y, params, v_params, g, e, Jg, Je, Hg, He, @dXH, @ddXH );