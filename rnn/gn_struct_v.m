function [Rparams Ry Rh Rt Rs Rdh Rdt Rds] = gn_struct_v( x, y, params, v_params, e, g, Je, Jg, Hg, dL, ddL, lm )
% Same as gauss_newton_v, but with a small modification to implement
% structural damping.  lm is lambda * mu, the strength of the structural
% damping coefficient.
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
Rdt = zeros(size(t)); % R_v[dL/dt], where L is the loss
Rds = zeros(size(s));
Rdh = zeros(size(t));
Rdtt = zeros(size(h0)); % Rdt one step in the future
for i = size(x,2):-1:1
    Rds(:,i) = Jg( s(:,i) )' * ( ddL( y(:,i), y_est(:,i) ) .* Ry(:,i) ) + ...
       tprod( Hg( s(:,i) ), [2 1 -1], Rs(:,i), -1 ) * dL( y(:,i), y_est(:,i) ); % R_v[dL/ds]
    RW_yh = RW_yh + Rds(:,i) * h(:,i)';
    Rb_y  = Rb_y  + Rds(:,i);
    
    Rdt(:,i) = Je( t(:,i) )' * ( W_hh' * Rdtt  +  W_yh' * Rds(:,i) + lm * Rt(:,i) );
    if i > 1
        RW_hh = RW_hh + Rdt(:,i) * h(:,i-1)';
    else
        RW_hh = RW_hh + Rdt(:,i) * h0';
    end
    RW_hx = RW_hx + Rdt(:,i) * x(:,i)';
    Rb_h  = Rb_h + Rdt(:,i);
    Rdtt = Rdt(:,i);
end

Rh0 = W_hh' * Rdtt;
Rparams = { Rh0, RW_hh, RW_hx, RW_yh, Rb_h, Rb_y };