function [ ret ] = make_double( x )
if isa(x, 'double')
    ret = x;
elseif strfind(x, ',') % try to parse as comma-separated list of doubles
    ret = str2double(strsplit(x, ','));
else
    ret = str2double(x);
end

