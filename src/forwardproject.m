function forward_model=forwardproject(gt,psf_ballistic,volume_n_px)


% Project forward patched
forward_model = zeros(length(gt), volume_n_px(1) * volume_n_px(2), 'single');
for ix=1:length(gt)
    img = project_forward_patched(gt{ix}, volume_n_px, psf_ballistic.Nnum, psf_ballistic.H);
    forward_model(ix, :) = img(:);
end
