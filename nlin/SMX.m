function y = SMX(x)
% softmax nonlinearity, useful for probabilistic models

y = exp(x)/sum(exp(x));