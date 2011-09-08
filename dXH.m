function dL = dXH(y,y_est)
% Gradient of the cross-entropy wrt y_est

dL = -y./y_est;