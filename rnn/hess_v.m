function [Rparams Ry Rh Rt Rs Rdh Rdt Rds] = hess_v( x, y, params, v_params, e, g, Je, Jg, He, Hg, dL, ddL )
% Calculates the product of the Hessian of the error wrt the weights with
% an arbitrary vector v according to the method of Pearlmutter 1994.
%
% v_params - cell array giving the direction along which we take the second
%            derivative
% He(i,j,k) - d^2e_i/dx_jdx_k
% Hg(i,j,k) - d^2g_i/dx_jdx_k
% ddL - second derivative of loss f'n wrt estimate (diagonal of Hessian)
%
% David Pfau, 2011

unbox;
for i = 1:length(paramnames)
    eval(['R' paramnames{i} ' = zeros(size(' paramnames{i} '));'])
    eval(['v' paramnames{i} ' = v_params{' int2str(i) '};'])
end

% Forward Pass
[y_est h t s] = rnn( x, params, e, g );

Ry = zeros(size(y));
Rh = zeros(size(W_hh,1),size(x,2));
Rs = zeros(size(Ry));
Rt = zeros(size(Rh)); % the R_v operator applied to the forward pass

Rht = vh0;
ht  = h0;
for i = 1:size(x,2)
    Rt(:,i) = W_hh * Rht + vW_hh * ht + vW_hx * x(:,i) + vb_h;
    Rh(:,i) = Je( t(:,i) ) * Rt(:,i);
    Rht = Rh(:,i);
    ht  =  h(:,i);
    
    Rs(:,i) = W_yh * Rht + vW_yh * ht + vb_y;
    Ry(:,i) = Jg( s(:,i) ) * Rs(:,i);
end

% Backward Pass
[~, dt, ds] = bptt( x, y, params, e, g, Je, Jg, dL, y_est, h, t, s );

Rdt = zeros(size(dt)); % R_v[dL/dt], where L is the loss
Rds = zeros(size(ds));
Rdh = zeros(size(dt));
dtt = zeros(size(h0)); % dt one step in the future
Rdtt = zeros(size(h0)); % Rdt one step in the future
for i = size(x,2):-1:1
    Rds(:,i) = Jg( s(:,i) )' * ( ddL( y(:,i), y_est(:,i) ) .* Ry(:,i) ) + ...
       tprod( Hg( s(:,i) ), [2 1 -1], Rs(:,i), -1 ) * dL( y(:,i), y_est(:,i) ); % R_v[dL/ds]
    RW_yh = RW_yh + Rds(:,i) * h(:,i)' + ds(:,i) * Rh(:,i)';
    Rb_y  = Rb_y  + Rds(:,i);
    
    Rdh(:,i) = ( W_hh' * Rdtt + vW_hh' * dtt ) + ( W_yh' * Rds(:,i) + vW_yh' * ds(:,i) );    
    Rdt(:,i) = Je( t(:,i) )' * Rdh(:,i) + tprod( He( t(:,i) ), [2 1 -1], Rt(:,i), -1 ) * ( W_yh' * ds(:,i) + W_hh' * dtt );
    if i > 1
        RW_hh = RW_hh + Rdt(:,i) * h(:,i-1)' + dt(:,i) * Rh(:,i-1)';
    else
        RW_hh = RW_hh + Rdt(:,i) * h0' + dt(:,i) * vh0';
    end
    RW_hx = RW_hx + Rdt(:,i) * x(:,i)';
    Rb_h  = Rb_h + Rdt(:,i);
    dtt = dt(:,i);
    Rdtt = Rdt(:,i);
end

Rh0 = vW_hh' * dtt + W_hh' * Rdtt;
Rparams = { Rh0, RW_hh, RW_hx, RW_yh, Rb_h, Rb_y };