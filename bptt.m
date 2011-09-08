function dparams = bptt( x, y, params, e, g, Je, Jg, dL )
% Calculates the derivatives of the error wrt weights and initial state in
% a recurrent neural net.  All arguments are the same as for rnn.m except:
%
% y - target output
% Je - Jacobian of hidden nonlinearity
% Jg - Jacobian of output nonlinearity (both diagonal if nonlinearity is
% pointwise)
% dL - gradient of loss wrt estimate
%
% David Pfau, 2011

% Forward pass
[y_est h t s] = rnn( x, params, e, g );

unbox;
for i = 1:length(paramnames)
    eval(['d' paramnames{i} ' = zeros(size(' paramnames{i} '));'])
end

% Backpropagation
dt = zeros(size(h0)); % dL/dt, where L is the loss
for i = size(x,2):-1:1
    ds  = Jg( s(:,i) )' * dL( y(:,i), y_est(:,i) ); % dL/ds
    dW_yh = dW_yh + ds * h(:,i)';
    db_y  = db_y  + ds;
    
    dt = Je( t(:,i) )' * ( W_yh' * ds + W_hh' * dt );
    if i > 1
        dW_hh = dW_hh + dt * h(:,i-1)';
    else
        dW_hh = dW_hh + dt * h0';
    end
    dW_hx = dW_hx + dt * x(:,i)';
    db_h = db_h + dt;
end

dh0 = W_hh' * dt;
dparams = { dh0, dW_hh, dW_hx, dW_yh, db_h, db_y };