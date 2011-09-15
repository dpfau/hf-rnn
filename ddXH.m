function ddL = ddXH(y,y_est)
% Diagonal of Hessian of the cross-entropy wrt y_est

ddL = y./y_est.^2;