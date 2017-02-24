function [ ret ] = is_double( x )
if isa(x, 'double')
    ret = 1;
else
    ret = isfloat(str2double(x));
end

