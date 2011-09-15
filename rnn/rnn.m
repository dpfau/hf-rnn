function [y h t s] = rnn( x, params, e, g )
% Given inputs and initial hidden state, calculates the output of a
% recurrent neural network
% x - input (one input per column)
% h0 - initial state vector
% W_hh - hidden-to-hidden weight matrix
% W_hx - input-to-hidden weight matrix
% W_yh - hidden-to-output weight matrix
% b_h - hidden bias
% b_y - output bias
% e - hidden nonlinearity
% g - output nonlinearity
% y - output (one output per column)
% h - hidden states
% t - linear input to hidden neuron, pre-squashing
% s - linear input to output neuron, pre-squashing
%
% David Pfau, 2011

unbox;

assert( size(x,1) == size(W_hx,2) );
assert( length(h0) == size(W_hh,1) );
assert( size(W_hh,1) == size(W_hh,2) );
assert( size(W_yh,2) == size(W_hh,2) );
assert( size(W_hx,1) == size(W_hh,2) );
assert( length(h0) == length(b_h) );
assert( length(b_y) == size(W_yh,1) );

y = zeros(length(b_y),size(x,2));
h = zeros(length(h0),size(x,2));

t = zeros(size(h));
s = zeros(size(y));

ht = h0;
for i = 1:size(x,2)
    t(:,i) = W_hx * x(:,i) + W_hh * ht + b_h;
    h(:,i) = e( t(:,i) );
    s(:,i) = W_yh * h(:,i) + b_y;
    y(:,i) = g( s(:,i) );
    ht = h(:,i);
end