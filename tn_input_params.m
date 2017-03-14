%% better miniscope
% Input.LFM_folder='/ssd_raid_4TB/tobias/X10_convert_test_2'; % no clear signal in stddev
% Input.x_offset = 637.2;
% Input.y_offset = 516.741;
% Input.dx = 19.76;

% Input.LFM_folder='/ssd_raid_4TB/tobias/2017-02-24_withcue_recording2/X12_20min_withcue_5HZ/tif/';
% Input.output_folder='/ssd_raid_4TB/tobias/2017-02-24_withcue_recording2/X12_20min_withcue_5HZ_sid';
% Input.output_name='X12_20min_withcue_5HZ';
% Input.x_offset = 642.0;
% Input.y_offset = 516.1;
% Input.dx = 19.76;

Input.LFM_folder='/ssd_raid_4TB/tobias/2017-02-24_withcue_recording2/X11_20min_withcue_5HZ/tif/';
Input.output_folder='/ssd_raid_4TB/tobias/2017-02-24_withcue_recording2/X11_20min_withcue_5HZ_sid';
Input.output_name='X11_20min_withcue_5HZ_';
Input.x_offset = 639.9;
Input.y_offset = 517.7;
Input.dx = 19.76;
Input.frames.start = 1;
Input.frames.steps = 10;
Input.frames.end = 1e6;

Input.LFM_folder='/ssd_raid_4TB/tobias/2017-02-24_withcue_recording2/X10_20min_withcue_5HZ/tif/';
Input.output_folder='/ssd_raid_4TB/tobias/2017-02-24_withcue_recording2/X10_20min_withcue_5HZ_sid';
Input.output_name='X10_20min_withcue_5HZ_';
Input.x_offset = 639.6;
Input.y_offset = 517.7;
Input.dx = 19.76;
Input.frames.start = 1;
Input.frames.steps = 3;
Input.frames.end = 7130; % sudden drop in brightness after this

%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_olympus_20x_05NA_water_117pm18mm_20FN_on_relay_stfica_pm100_from-100_to100_zspacing4_Nnum11_lambda520_OSR3_normed';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_05NA_M8P66_water_from-50_to250_zspacing4_Nnum15_lambda520_OSR3.mat';


Input.rank=8; % If Input.rank==0 SID classic instead of SID_nmf
Input.step=1;
Input.step_=3;
Input.bg_iter=2;
Input.rectify=1;
Input.Junk_size=1000;
Input.bg_sub=1;
Input.prime=10000;
Input.prime_=4800;
Input.gpu_ids=[1 2];
Input.num_iter=4;
Input.native_focal_plane=12;
Input.thres = 10;

Input.recon_opts.lambda = [0, 0, 10];
Input.recon_opts.lambda_ = 0.2;
Input.recon_opts.p = 1;
Input.recon_opts.maxIter = 8;
Input.recon_opts.mode = 'TV';
Input.recon_opts.whichSolver = 'fast_nnls';

%% nothing in stddev
Input.LFM_folder='/ssd_raid_4TB/tobias/X11_post_screwhole';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_olympus_20x_05NA_water_117pm18mm_20FN_on_relay_stfica_pm100_from-100_to100_zspacing4_Nnum11_lambda520_OSR3_normed';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_05NA_M8P66_water_from-50_to250_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.output_folder='/ssd_raid_4TB/tobias/X11_post_screwhole_sid';
Input.output_name='X11_post_screwhole4_';
Input.x_offset = 637.2; %TODO check
Input.y_offset = 516.741;
Input.dx = 19.76;

%%


%%
Input.LFM_folder='/ssd_raid_4TB/tobias/2016-03-04_m12_cyto-r_hippo/pos0_zm085_2min_5fps_corr-depth_1/Pos0';
Input.output_folder='/ssd_raid_4TB/tobias/2016-03-04_m12_cyto-r_hippo/pos0_zm085_2min_5fps_corr-depth_1_sid';
Input.output_name='160304_m12_cyto-r_hippo_zm085';
Input.x_offset = 1283.6;
Input.y_offset = 1078.5;
Input.dx = 17.725;
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-100_to100_zspacing4_Nnum15_lambda520_OSR3_pdfnorm.mat';

%%
Input.LFM_folder='/ssd_raid_4TB/tobias/2016-03-04_m12_cyto-r_hippo/pos0_zm100_2min_5fps_corr-depth_2/Pos0/';
Input.output_folder='/ssd_raid_4TB/tobias/2016-03-04_m12_cyto-r_hippo/pos0_zm100_2min_5fps_corr-depth_1_sid';
Input.output_name='160304_m12_cyto-r_hippo_zm100';
Input.x_offset = 1283.6;
Input.y_offset = 1078.5;
Input.dx = 17.725;
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-100_to100_zspacing4_Nnum15_lambda520_OSR3_pdfnorm.mat';

%%
Input.LFM_folder='/ssd_raid_4TB/tobias/2015-09-16_APmouse13_cytoRcamp/pos1_05NA_z200_200ms_5min__1/pos1_05NA_z200_200ms_5min/';
Input.output_folder='/ssd_raid_4TB/tobias/2015-09-16_APmouse13_cytoRcamp/pos1_05NA_z200_200ms_5min__1_sid';
Input.output_name='pos1_05NA_z200_200ms_5min__1';
Input.x_offset = 1095.9;
Input.y_offset = 1063.3;
Input.dx = 22.595;
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_olympus_20x_05NA_water__20FN__on_scientifica_from-100_to100_zspacing4_Nnum11_lambda520_OSR3.mat';
Input.rank=30; % If Input.rank==0 SID classic instead of SID_nmf
Input.step=1;
Input.step_=3;
Input.bg_iter=2;
Input.rectify=1;
Input.Junk_size=1000;
Input.bg_sub=1;
Input.prime=40000;
Input.prime_=4800;
Input.gpu_ids=[4 5];
Input.num_iter=4;
Input.native_focal_plane=12;
Input.thres = 10;

Input.recon_opts.lambda = [0, 0, 10];
Input.recon_opts.lambda_ = 0;
Input.recon_opts.p = 1;
Input.recon_opts.maxIter = 8;
Input.recon_opts.mode = 'TV';
Input.recon_opts.whichSolver = 'fast_nnls';