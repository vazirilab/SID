%%
files = dir('../src/*.m');
%files = files(1:3);
options = struct(...
    'format', 'html', ...
    'outputDir', '../docs/published_m_files/', ...
    'evalCode', false);
for file = files'
    publish(file.name, options);
end