%%
%recon_opts.infile = '~/vazirilab_medium_data/joint_projects/lfm_general/2017-02-28_LFM-realign_20x05NApm/USAF_above_nfp.tif';
recon_opts.infile = '~/vazirilab_medium_data/tmp/fish1_LQ/img_000003720_Default_000.tif';
recon_opts.offset_x = 1020.9;
recon_opts.offset_y = 1027.2;
recon_opts.dx = 22.880000;

% recon_opts.infile = '~/vazirilab_medium_data/tmp/compare_pm_fish/img_000000000_Default_000.tif';
% recon_opts.offset_x = 1019.9;
% recon_opts.offset_y = 1025.3;
% recon_opts.dx = 23.260000;
% 
% recon_opts.infile = '~/vazirilab_medium_data/tmp/2016-03-02_fish_friederike/img_000000000_Default_000.tif';
% recon_opts.offset_x = 1021.5;
% recon_opts.offset_y = 1025.5;
% recon_opts.dx = 23.3;
% 
recon_opts.psf = '/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_olympus_20x_05NA_water_117pm18mm_20FN_on_relay_stfica_pm100_from-100_to100_zspacing4_Nnum11_lambda520_OSR3_normed.mat';

recon_opts.p = 2;
recon_opts.maxIter = 8;
recon_opts.mode = 'TV';
recon_opts.lambda = [0, 0, 10];
recon_opts.lambda_ = 0.1;

% recon_opts.p = 1;
% recon_opts.maxIter = 8;
% recon_opts.mode = 'basic';
% recon_opts.lambda = 0;
% recon_opts.whichSolver = 'ISRA';

recon_opts.gpu_ids = [5];
do_crop = 1;
crop_thresh_coord_x = 0.7;
crop_thresh_coord_y = 0.7;

%%
psf = load(recon_opts.psf);

%%
in_raw = read_tiff_stack(recon_opts.infile);
clear tmp;
tmp.LFmovie(:, :, 1) = ImageRect(double(in_raw), recon_opts.offset_x, recon_opts.offset_x, recon_opts.dx, size(psf.H, 3), 0, 400/recon_opts.dx, 400/recon_opts.dx, 400/recon_opts.dx, 300/recon_opts.dx);

%% make smooth mask that excludes background (non-brain areas)
output.std_image = tmp.LFmovie(:, :, 1);
if do_crop
    disp('Finding crop space');
    sub_image = output.std_image(ceil(crop_thresh_coord_x * size(output.std_image,1)):end, ceil(crop_thresh_coord_y * size(output.std_image,2)):end);
    sub_image = output.std_image - mean(sub_image(:)) - 2*std(sub_image(:));
    sub_image(sub_image<0) = 0;
    beads = bwconncomp(sub_image>0);
    for kk = 1:beads.NumObjects
        if numel(beads.PixelIdxList{kk}) < 8
            sub_image(beads.PixelIdxList{kk}) = 0;
        end
    end
    h = fspecial('average', 2*psf.Nnum);
    sub_image = conv2(sub_image,h,'same');
    output.idx = find(sub_image>0);
else
    sub_image = output.std_image * 0 + 1;
end
tmp.LFmovie(:, :, 1) = tmp.LFmovie(:, :, 1) .* (sub_image > 0);

%%
figure; imagesc(tmp.LFmovie(:, :, 1)); axis image

%%
out.recon = reconstruction_sparse(tmp, psf, recon_opts);

%%
[fp, fn, fe] = fileparts(recon_opts.infile);
recon_opts.outdir = [fp '/recon_tv'];
mkdir(recon_opts.outdir);
out.recon_uint16 = uint16((out.recon - min(out.recon(:))) / (max(out.recon(:)) - min(out.recon(:))) * 65535);
write_tiff_stack(fullfile(recon_opts.outdir, [fn fe]), out.recon_uint16);
disp('Done writing outfile');
gpuDevice([]);


