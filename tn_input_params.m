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

%%
optional_arg = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/2017-04-26_m29/tif/';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/data/FM_miniLFM_recordings/m29/042617_miniLFM_sid/';
Input.output_name='2017-04-26_m29_wd200_';
Input.x_offset = 638.2;
Input.y_offset = 522.9;
Input.dx = 19.65;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 7;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
optional_args = struct;
Input.optimize_kernel = 0;

%%
optional_arg = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/2017-05-01_m30_rec2/';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-05-01_m30_rec2';
Input.output_name='2017-05-01_m30_rec2_wd200';
Input.x_offset = 639.4;
Input.y_offset = 500.4;
Input.dx = 19.565;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 7;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
optional_args = struct;
Input.optimize_kernel = 0;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/M24_LFM_WD350_Depth200um_1/tif/';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/M24_LFM_WD350_Depth200um_1_nomotion-only_psfm450p200';
Input.output_name='M24_LFM_WD350_Depth200um_1';
Input.x_offset = 641.0;
Input.y_offset = 496.8;
Input.dx = 19.755;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 7;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;
% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[5 2];

Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;

Input.nnmf_opts.max_iter = 300;
Input.nnmf_opts.lambda_t = 0;
Input.nnmf_opts.lambda_s = 100;

Input.frames_for_model_optimization.start = 500;
Input.frames_for_model_optimization.end = 2000;
Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/M24_LFM_WD350_Depth200um_Overexposure/tif/';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/M24_LFM_WD350_Depth200um_Overexposure_psfm450p200';
Input.output_name='M24_LFM_WD350_Depth200um_Overexposure';
Input.x_offset = 640.4;
Input.y_offset = 496.8;
Input.dx = 19.755;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 1;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 20;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='TV';
Input.recon_opts.lambda=[ 0, 0, 10];
Input.recon_opts.lambda_=0.1;
Input.recon_opts.form='gaussian';
Input.recon_opts.rad=[5 2];
Input.mask_file = '~/vazirilab_medium_data/joint_projects/miniscope/analyses/M24_LFM_WD350_Depth200um_Overexposure/mask.tif';
Input.nnmf_opts.lambda_s = 200;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-05-01_m33_1/tif/';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-05-01_m33_1';
Input.output_name='2017-05-01_m33';
Input.x_offset = 638.3;
Input.y_offset = 495.7;
Input.dx = 19.755;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_m450_from-450_to-200_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_m450_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 7;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];

Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;

Input.nnmf_opts.lambda_s = 100;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/M24_LFM_WD200_Depth200um/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/M24_LFM_WD200_Depth200um';
Input.output_name='M24_LFM_WD200_Depth200um';
Input.x_offset = 640.1;
Input.y_offset = 496.9;
Input.dx = 19.755;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 7;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];

Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;

Input.nnmf_opts.lambda_s = 100;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/M24_LFM_WD200_Depth000um/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/M24_LFM_WD200_Depth000um';
Input.output_name='M24_LFM_WD200_Depth000um';
Input.x_offset = 640.3;
Input.y_offset = 496.9;
Input.dx = 19.755;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 7;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];

Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;

Input.nnmf_opts.lambda_s = 100;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-09_c01_1/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-09_c01_1';
Input.output_name='2017-06-09_c01_1';
Input.x_offset = 640.2;
Input.y_offset = 498.9;
Input.dx = 19.630;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 7;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='TV';
Input.recon_opts.lambda=[ 0, 0, 10];
Input.recon_opts.lambda_=0.1;
Input.recon_opts.form='gaussian';
Input.recon_opts.rad=[3 1];
% 
% Input.recon_opts = struct;
% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='basic';
% Input.recon_opts.lambda=0;
% Input.recon_opts.lambda_=0.0;

Input.nnmf_opts.lambda_s = 100;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-09_c04_2/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-09_c04_2';
Input.output_name='2017-06-09_c04_2';
Input.x_offset = 642.2;
Input.y_offset = 497.1;
Input.dx = 19.630;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='TV';
Input.recon_opts.lambda=[ 0, 0, 10];
Input.recon_opts.lambda_=0.1;
Input.recon_opts.form='gaussian';
Input.recon_opts.rad=[3 1];
% 
% Input.recon_opts = struct;
% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='basic';
% Input.recon_opts.lambda=0;
% Input.recon_opts.lambda_=0.0;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-09_c04_3/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-09_c04_3';
Input.output_name='2017-06-09_c04_3';
Input.x_offset = 642.2;
Input.y_offset = 497.1;
Input.dx = 19.630;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [2,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='TV';
Input.recon_opts.lambda=[ 0, 0, 10];
Input.recon_opts.lambda_=0.1;
Input.recon_opts.form='gaussian';
Input.recon_opts.rad=[3 1];
% 
% Input.recon_opts = struct;
% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='basic';
% Input.recon_opts.lambda=0;
% Input.recon_opts.lambda_=0.0;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-12_c05_lfm_1/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-12_c05_lfm_1';
Input.output_name='2017-06-12_c05_lfm_1';
Input.x_offset = 644.6;
Input.y_offset = 498.4;
Input.dx = 19.630;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='TV';
Input.recon_opts.lambda=[ 0, 0, 10];
Input.recon_opts.lambda_=0.1;
Input.recon_opts.form='gaussian';
Input.recon_opts.rad=[3 1];
% 
% Input.recon_opts = struct;
% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='basic';
% Input.recon_opts.lambda=0;
% Input.recon_opts.lambda_=0.0;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-12_c05_lfm_2/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-12_c05_lfm_2__nomotion-only';
Input.output_name='2017-06-12_c05_lfm_2';
Input.x_offset = 644.6;
Input.y_offset = 498.4;
Input.dx = 19.630;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
Input.frames_for_model_optimization.start = 625;
Input.frames_for_model_optimization.end = 2000;
Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-12_c09_lfm_2/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-12_c09_lfm_2';
Input.output_name='2017-06-12_c09_lfm_2';
Input.x_offset = 643.9;
Input.y_offset = 496.4;
Input.dx = 19.630;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-15_c07_lfm_1/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-15_c07_lfm_1';
Input.output_name='2017-06-15_c07_lfm_1';
Input.x_offset = 642.9;
Input.y_offset = 498.8;
Input.dx = 19.630;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-16_c02_lfm_1/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-16_c02_lfm_1';
Input.output_name='2017-06-16_c02_lfm_1';
Input.x_offset = 640.5;
Input.y_offset = 519.6;
Input.dx = 19.64;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-06-16_c07_lfm_3/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-06-16_c07_lfm_3';
Input.output_name='2017-06-16_c07_lfm_3';
Input.x_offset = 643.6;
Input.y_offset = 517.8;
Input.dx = 19.64;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-09-28_lintrack_m58_1/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-09-28_lintrack_m58_1';
Input.output_name='2017-09-28_lintrack_m58_1';
Input.x_offset = 646.9;
Input.y_offset = 514.6;
Input.dx = 19.64;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-09-28_lintrack_m56_1/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-09-28_lintrack_m56_1';
Input.output_name='2017-09-28_lintrack_m56_1';
Input.x_offset = 646.9;
Input.y_offset = 514.6;
Input.dx = 19.64;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-09-28_lintrack_m57_1/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-09-28_lintrack_m57_1';
Input.output_name='2017-09-28_lintrack_m57_1';
Input.x_offset = 647.6;
Input.y_offset = 517.4;
Input.dx = 19.64;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-09-14_m65_nucl-g/m65_zm000_30fps__1/Pos0';
Input.output_folder='~/vazirilab_medium_data/joint_projects/sid/analyses/2017-09-14_m65_nucl-g/m65_zm000_30fps__1/';
Input.output_name='2017-09-14_lintrack_m65_zm000';
Input.x_offset = 1283.0;
Input.y_offset = 1082.6;
Input.dx = 17.75;
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-100_to100_zspacing4_Nnum15_lambda520_OSR3_pdfnorm.mat';
Input.do_crop = 0;
Input.detrend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [1,2]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-09-14_m65_nucl-g/m65_zm150_30fps__1/Pos0';
Input.output_folder='~/vazirilab_medium_data/joint_projects/sid/analyses/2017-09-14_m65_nucl-g/m65_zm150_30fps__1/';
Input.output_name='2017-09-14_lintrack_m65_zm150';
Input.x_offset = 1283.0;
Input.y_offset = 1082.6;
Input.dx = 17.75;
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-100_to100_zspacing4_Nnum15_lambda520_OSR3_pdfnorm.mat';
Input.do_crop = 0;
Input.detrend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [5]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-09-14_m65_nucl-g/m65_zm250_30fps__1/Pos0';
Input.output_folder='~/vazirilab_medium_data/joint_projects/sid/analyses/2017-09-14_m65_nucl-g/m65_zm250_30fps__1/';
Input.output_name='2017-09-14_lintrack_m65_zm250';
Input.x_offset = 1283.0;
Input.y_offset = 1082.6;
Input.dx = 17.69;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-100_to100_zspacing4_Nnum15_lambda520_OSR3_pdfnorm.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-252_to-100_zspacing4_Nnum15_lambda520_OSR3.mat';

%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-252_to_4_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/dev/shm/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-252_to_4_zspacing4_Nnum15_lambda520_OSR3_uncompressed.mat';
Input.psf_cache_dir = '';

Input.do_crop = 0;
Input.crop_border_microlenses = [0 0 0 floor(400/15)];

Input.detrend = true;

Input.rank = 30;
Input.nnmf_opts.lambda_t = 0;
Input.nnmf_opts.lambda_s = 10;
Input.nnmf_opts.lambda_orth = 4; % maybe try 40
Input.nnmf_opts.max_iter = 300;

Input.thres = 20;
Input.gpu_ids = [4]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-09-14_m65_nucl-g/m65_zm350_30fps__1/Pos0';
Input.output_folder='~/vazirilab_medium_data/joint_projects/sid/analyses/2017-09-14_m65_nucl-g/m65_zm350_30fps__1/';
Input.output_name='2017-09-14_lintrack_m65_zm350';
Input.x_offset = 1283.0;
Input.y_offset = 1082.6;
Input.dx = 17.75;


Input.psf_filename_ballistic='/dev/shm/PSFmatrix_nikon_16x_08NA_water__10FN__on_scientifica__touching_circles_from-252_to_4_zspacing4_Nnum15_lambda520_OSR3_uncompressed.mat';
Input.psf_cache_dir = '';

Input.do_crop = 0;
Input.crop_border_microlenses = [0 0 0 floor(400/15)];

Input.detrend = true;

Input.rank = 30;
Input.nnmf_opts.lambda_t = 0;
Input.nnmf_opts.lambda_s = 10;
Input.nnmf_opts.lambda_orth = 4; % maybe try 40
Input.nnmf_opts.max_iter = 300;

Input.thres = 20;
Input.gpu_ids = [4]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-10-12_m57/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-10-12_m57';
Input.output_name='2017-10-12_m57';
Input.x_offset = 643.0;
Input.y_offset = 520.3;
Input.dx = 19.695;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.do_crop = 0;
Input.detrend = false;
Input.de_trend = true;
Input.rank = 10;
Input.nnmf_opts.lambda_s = 10;
Input.thres = 20;
Input.gpu_ids = [2]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;
% Input.frames_for_model_optimization.start = 625;
% Input.frames_for_model_optimization.end = 2000;
% Input.frames_for_model_optimization.step = 1;

%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-10-12_m56/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-10-12_m56';
Input.output_name='2017-10-12_m56';
Input.x_offset = 642.8;
Input.y_offset = 516.9;
Input.dx = 19.64;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';

Input.do_crop = 0;
Input.detrend = true;

Input.rank = 20;
Input.nnmf_opts.lambda_t = 0;
Input.nnmf_opts.lambda_s = 10;
Input.nnmf_opts.lambda_orth = 4; % maybe try 40
Input.nnmf_opts.max_iter = 300;

Input.thres = 20;
Input.gpu_ids = [4]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;


%%
optional_args = struct;
Input.LFM_folder='/ssd_raid_4TB/tobias/data_cache/2017-10-12_m58_2/tif';
Input.output_folder='~/vazirilab_medium_data/joint_projects/miniscope/analyses/2017-10-12_m58_2';
Input.output_name='2017-10-12_m58_2';
Input.x_offset = 641.0;
Input.y_offset = 517.8;
Input.dx = 19.695;
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_water_from-266_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p360_from-50_to360_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_p400_from50_to400_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M5P97_w_pm100_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_w_p300_from-320_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd266_0435NA_M8P95_p300_from-100_to300_zspacing4_Nnum15_lambda520_OSR3.mat';
%Input.psf_filename_ballistic='/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_miniscope_wd350_0493NA_M7P98_w_from-450_to200_zspacing4_Nnum15_lambda520_OSR3.mat';

Input.do_crop = 0;
Input.detrend = true;

Input.rank = 20;
Input.nnmf_opts.lambda_t = 0;
Input.nnmf_opts.lambda_s = 10;
Input.nnmf_opts.lambda_orth = 4; % maybe try 40
Input.nnmf_opts.max_iter = 300;

Input.thres = 20;
Input.gpu_ids = [4]; %1-based! so 1,2,4,5 are valid
Input.optimize_kernel = 0;

% Input.recon_opts.p=2;
% Input.recon_opts.maxIter=8;
% Input.recon_opts.mode='TV';
% Input.recon_opts.lambda=[ 0, 0, 10];
% Input.recon_opts.lambda_=0.1;
% Input.recon_opts.form='gaussian';
% Input.recon_opts.rad=[3 1];
% 
Input.recon_opts = struct;
Input.recon_opts.p=2;
Input.recon_opts.maxIter=8;
Input.recon_opts.mode='basic';
Input.recon_opts.lambda=0;
Input.recon_opts.lambda_=0.0;

%% C. elegans
indir = '~/vazirilab_medium_data/tobias_noebauer/data_menachem_lfm/2018-05-18_menachem_celegans/1088-WT-1-Tstim_1/Pos0/';
outdir = '/ssd_raid_4TB/tobias/menachem_celegans_sid/1088-WT-1-Tstim_1/';
psffile = '/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_zeiss_63x_09NA_water__20FN__on_zeiss_from-18_to18_zspacing2_Nnum15_lambda520_OSR3.mat';
x_offset = 1288.4;
y_offset = 1093.8;
dx = 22.13;

optional_args = struct;
optional_args.frames.step = 2;
optional_args.delta = 20;  % detrending sliding window
optional_args.axial = 12.6;  % on F#20 MLA, ML pitch is 150um. M=63x. Nnum=15. So 1 px is 0.159 um laterally, and 2 um axially
optional_args.neur_rad = 40;
optional_args.native_focal_plane = 10;
optional_args.SID_output_name = '1088-WT-1-Tstim_1';
optional_args.gpu_ids = [4 5];
optional_args.crop_border_microlenses = [0 0 33 29];
optional_args.nnmf_opts.xval.enable = false;
optional_args.ts_extract_chunk_size = 200;

%% C. elegans
indir = '/ssd_raid_4TB/tobias/data_cache_celegans/1088-glt1-1-Tstim_1/Pos0/';
outdir = '/ssd_raid_4TB/tobias/menachem_celegans_sid/1088-glt1-1-Tstim_1/';
psffile = '/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_zeiss_63x_09NA_water__20FN__on_zeiss_from-18_to18_zspacing2_Nnum15_lambda520_OSR3.mat';
x_offset = 1288.4;
y_offset = 1093.8;
dx = 22.13;

optional_args = struct;
optional_args.frames.start = 1;
optional_args.frames.step = 2;
optional_args.frames.end = inf;
optional_args.frames.mean = true;
optional_args.delta = 20;  % detrending sliding window
optional_args.axial = 12.6;  % on F#20 MLA, ML pitch is 150um. M=63x. Nnum=15. So 1 px is 0.159 um laterally, and 2 um axially
optional_args.neur_rad = 40;
optional_args.native_focal_plane = 10;
optional_args.SID_output_name = '1088-glt1-1-Tstim_1';
optional_args.gpu_ids = [4 5];
optional_args.crop_params = [0.4 0.8];
optional_args.crop_border_microlenses = [43 19 0 0 ];
%optional_args.nnmf_opts.xval_enable = false;
optional_args.ts_extract_chunk_size = 200;

%% C. elegans, new param struct
config_in = struct;

config_in.indir = '/ssd_raid_4TB/tobias/data_cache_celegans/1088-glt1-5-Tstim_2/Pos0';
config_in.outdir = '/ssd_raid_4TB/tobias/menachem_celegans_sid/1088-glt1-5-Tstim_2/';
config_in.psffile = '/ssd_raid_4TB/lfm_reconstruction_PSFs/PSFmatrix_zeiss_63x_09NA_water__20FN__on_zeiss_from-18_to18_zspacing2_Nnum15_lambda520_OSR3.mat';
config_in.x_offset = 1288.4;
config_in.y_offset = 1093.8;
config_in.dx = 22.13;

config_in.frames.start = 1;
config_in.frames.step = 3;
config_in.frames.end = inf;

config_in.axial = 12.6;  % on F#20 MLA, ML pitch is 150um. M=63x. Nnum=15. So 1 px is 0.159 um laterally, and 2 um axially
config_in.neur_rad = 25;
config_in.delta = 100;
config_in.native_focal_plane = 10;
config_in.gpu_ids = [2 4 5];
config_in.crop_border_microlenses = floor([840 2560-2000 900 0] / config_in.dx);
config_in.crop_params = [0.5 0.8];

%Input.optimize_kernel = true;
%Input.recon_opts.lamb_L1 = 1;
%Input.filter = true;

Input.optimize_kernel = false;
Input.recon_opts.lamb_L1 = 0.1;
Input.recon_opts.lamb_TV_L2 = [0.1 0.1 4];
Input.filter = true;

config_in.recon_final_spatial_filters = false;

%%
main_nnmf_SID(config_in)


%%
%output.segmm_sav = output.segmm;
for i=1:numel(output.segmm)
    tmp = output.segmm{i};
    tmp(:,:,60:end) = 0;
    output.segmm{i} = tmp;
end

%%
save(fullfile(Input.output_folder, 'checkpoint_post-nmf-recon.mat'), 'Input', 'output');

%% quick psf plot
zixs = [1 13 26 39 51];
for i=1:numel(zixs) 
    zix = zixs(i);
    x0 = 165 - 60;
    x1 = 165 + 60;
    figure; imagesc(psf_ballistic.H(x0:x1,x0:x1,7,7,zix), [0 0.012]); axis image; colorbar
    title(num2str(psf_ballistic.x3objspace(zix) * 1e6));
    print(['~/vazirilab_medium_data/tmp/psf_wd266_m9p95_' num2str(zix) '.pdf'], '-dpdf', '-r300');
end

%%
write_tiff_stack('/tmp/nmf_recon_tmp.tif', uint16(mat2gray(output.recon{4}) * 65535)); !/opt/Fiji.app/ImageJ-linux64 /tmp/nmf_recon_tmp.tif

%%
figure; 
imagesc(reshape(output.S(1,:), size(output.std_image)));  
set(gca, 'xtick', 0.5:15:size(output.std_image, 2)); 
set(gca, 'ytick', 0.5:15:size(output.std_image, 1));
set(gca, 'GridColor', 'w');
set(gca, 'GridAlpha', 0.7);
grid on;
axis image; 
axis ij; 
colorbar;
%savefig(filename);



%%
img = reshape(output.S(1,:), size(output.std_image));
figure; imagesc(img(30*15+2:34*15+1, 30*15+2:34*15+1));  grid on; xticks(1:15:end); axis image; axis ij; colorbar;