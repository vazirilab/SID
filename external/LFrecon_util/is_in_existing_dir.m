function [ ret ] = is_in_existing_dir( filename )
[pathstr,name,ext] = fileparts(filename);
if exist(pathstr, 'dir')
    ret = 1;
else
    ret = 0;
end
end

