% https://www.mathworks.com/matlabcentral/answers/282820-programmatically-run-and-export-live-script
% java call seems to require absolute paths
[scriptpath, ~, ~] = fileparts(mfilename('fullpath'));
inpath = fullfile(scriptpath, '../demos/SIDdemo.mlx') %#ok<*NOPTS>
outpath = fullfile(scriptpath, '../docs/SIDdemo.html')
matlab.internal.liveeditor.openAndConvert(inpath, outpath)