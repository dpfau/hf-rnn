function y = dSMX(x)
% Jacobian of the softmax nonlinearity

y = diag(exp(x)/sum(exp(x))) - exp(x)*exp(x)'/sum(exp(x))^2;