function [dparams dt ds] = bptt( x, y, params, e, g, Je, Jg, dL, y_est, h, t, s )
% Calculates the derivatives of the error wrt weights and initial state in
% a recurrent neural net.  All arguments are the same as for rnn.m except:
%
% y - target output
% Je - Jacobian of hidden nonlinearity
% Jg - Jacobian of output nonlinearity (both diagonal if nonlinearity is
% pointwise)
% dL - gradient of loss wrt estimate
%
% dparams - cell array of error gradients, in same order as params
% dt - dL/dt, that is the error propagated backward from the hidden units
% ds - dL/ds
%
% David Pfau, 2011

% Forward pass
if nargin > 12
    [y_est h t s] = rnn( x, params, e, g );
end

unbox;
for i = 1:length(paramnames)
    eval(['d' paramnames{i} ' = zeros(size(' paramnames{i} '));'])
end

% Backpropagation
dt = zeros(size(t)); % dL/dt over all time steps, where L is the loss
ds = zeros(size(s));
dtt = zeros(size(h0)); % dL/dt at the current time step
for i = size(x,2):-1:1
    ds(:,i)  = Jg( s(:,i) )' * dL( y(:,i), y_est(:,i) ); % dL/ds
    dW_yh = dW_yh + ds * h(:,i)';
    db_y  = db_y  + ds;
    
    dt(:,i) = Je( t(:,i) )' * ( W_yh' * ds(:,i) + W_hh' * dtt );
    dtt = dt(:,i);
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