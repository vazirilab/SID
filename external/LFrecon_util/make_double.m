function [ ret ] = make_double( x )
if isa(x, 'double')
    ret = x;
elseif contains(x, ',') % try to parse as comma-separated list of doubles, optionally with brackets
    x = strip(x, '[');
    x = strip(x, ']');
    ret = str2double(strsplit(x, ','));
elseif contains(strip(x), ' ') % try to parse as space list of doubles, optionally with brackets
    x = strip(x, '[');
    x = strip(x, ']');
    x = strip(x);
    ret = str2double(strsplit(x, ' '));
else
    ret = str2double(x);
end

