% Numerically verify the correctness of my implementation of
% backpropagation in time.  David Pfau, 2011

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
e = @(x) exp(x)/sum(exp(x)); % softmax
Je = @(x) diag(exp(x)/sum(exp(x))) - exp(x)*exp(x)'/sum(exp(x))^2;

% g = @(x) x;
% Jg = @(x) eye(length(x));
% e = @(x) x;
% Je = @(x) eye(length(x));

[y_est,~,~,~] = rnn( x, params, g, e );
L = XH(y,y_est);

diff_params = cell(6,1); % the numerically computed gradients
for i = 1:length(diff_params)
    diff_params{i} = zeros(size(params{i}));
end

dparams = bptt( x, y, params, g, e, Jg, Je, @dXH ); % the exact gradients

eps = 1e-5;
for t = 1:length(params)
    for i = 1:size(params{t},1)
        for j = 1:size(params{t},2)
            params{t}(i,j) = params{t}(i,j) + eps;
            [y_est,~,~,~] = rnn( x, params, g, e );
            diff_params{t}(i,j) = (XH(y,y_est) - L)/eps;
            params{t}(i,j) = params{t}(i,j) - eps;
        end
    end
end