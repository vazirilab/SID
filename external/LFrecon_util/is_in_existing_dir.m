function [ ret ] = is_in_existing_dir( filename )
[pathstr,~,~] = fileparts(strip(filename, 'right', '/'));
if exist(pathstr, 'dir')
    ret = 1;
else
    ret = 0;
end
end

