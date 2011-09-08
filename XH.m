function L = XH(y,y_est)
% Cross-entropy error function

logy = log(y_est);
logy(isinf(logy)) = 0;
L = -sum(sum(y.*logy));