function params1 = conj_grad( params0, grad, hess_v, p0 )
% Linear conjugate gradient ascent, for the inner loop of HF optimization
% Minimizes 1/2*x'*A*x - b'*x
% params0 - cell array with initial position of x
% grad - cell array with b
% hess_v - function handle that returns A*x given x
% p0 - cell array with initial search direction
% params1 - final value of x

params1 = params0;
r = p0;
p = p0;
while 1 % remember to add cutoff condition!
    
end