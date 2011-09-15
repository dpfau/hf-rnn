% Test the Gauss-Newton approximation to the Hessian against the true
% Hessian to assure positive definiteness.  David Pfau, 2011

n = 100;
m = 10;
k = 10;
t = 10;

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

while 1
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
    
    v_params = cell(6,1); % the vector with which we multiply the Hessian
    for i = 1:length(params)
        v_params{i} = randn(size(params{i}));
    end
    
    Hparams  = hess_v( x, y, params, v_params, g, e, Jg, Je, Hg, He, @dXH, @ddXH );
    GNparams = gauss_newton_v( x, y, params, v_params, g, e, Jg, Je, He, @dXH, @ddXH );
    
    Hdot = 0;
    GNdot = 0;
    for i = 1:length(params)
        Hdot  = Hdot  + v_params{i}(:)'*Hparams{i}(:);
        GNdot = GNdot + v_params{i}(:)'*GNparams{i}(:);
    end
    fprintf('True: %d, GN approx: %d\n', celldot(v_params,Hparams), celldot(v_params,GNparams));
end