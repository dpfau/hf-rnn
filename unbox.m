% assign variable names to params in cell array

paramnames = { 'h0', 'W_hh', 'W_hx', 'W_yh', 'b_h', 'b_y' };

for i = 1:6
    eval([paramnames{i} ' = params{' int2str(i) '};']);
end