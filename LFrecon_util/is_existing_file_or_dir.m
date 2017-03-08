function [ ret ] = is_existing_file_or_dir(filename, allowed_extensions)
% Tests if filename is an existing file, with the extension being in the
% given cell array of strings allowed_extensions. If second argument is not
% there, file extension is not checked.

[pathstr,name,ext] = fileparts(filename);
ext = ext(2:end); % remove dot

if nargin < 2
    allowed_extensions = ext;
end

if exist(filename, 'file') == 2 && any(strcmpi(ext, allowed_extensions))
    ret = 1;
elseif  exist(filename, 'dir')
    ret = 1;
else
    ret = 0;
end

