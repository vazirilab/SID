function write_tiff_stack( save_fn, volume )
save_fn = fullfile(save_fn)
imwrite(volume(:,:,1), save_fn)
for k = 2:size(volume,3)
    imwrite(volume(:,:,k), save_fn, 'WriteMode', 'append');
end
end

