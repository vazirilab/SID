function [ ret ] = is_positive_integer_or_zero( x )
if ~isa(x, 'double')
    x = str2double(x);
end
ret = x == inf || (x >= 0 && ~mod(x, 1));
end