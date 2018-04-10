% 2-17-03-15
% Rawdata_name = 'Raw_2016-01-30_QL_FishOpCond_026_learnerRT_fish2_session2';
% Rawdata_folder = '/rugpfs/fs0/vzri_lab/scratch/xfer/operant_conditioning_vienna/Qian_Lin/Raw/';
% Input.x_offset=1022.2;
% Input.y_offset=1023.2;
% Input.dx=23.3;

% % 2017-04-02
% % 2017-10-05
% Rawdata_name = 'Raw_2016-03-25_QL_FishOpCond_055_openloop_fish1_blk2';
% Rawdata_folder = '/rugpfs/fs0/vzri_lab/scratch/xfer/operant_conditioning_vienna/Qian_Lin/Raw/';
% Input.x_offset=1022.0;
% Input.y_offset=1025.6;
% Input.dx=23.3;

% 2017-10-25
Rawdata_name = 'Raw_2016-03-25_QL_FishOpCond_057_openloop_fish2_blk2';
Rawdata_folder = '/rugpfs/fs0/vzri_lab/scratch/xfer/operant_conditioning_vienna/Qian_Lin/Raw/';
Input.x_offset=1022.0;
Input.y_offset=1025.6;
Input.dx=23.3;

Input.LFM_folder=[Rawdata_folder '/' Rawdata_name '/'];
Input.psf_filename_ballistic='/ru-auth/local/home/qlin/lfm_reconstruction_PSFs/PSFmatrix_olympus_20x_05NA_water_117pm18mm_20FN_on_relay_stfica_pm100_from-100_to100_zspacing4_Nnum11_lambda520_OSR3_normed.mat';
Input.output_folder='/rugpfs/fs0/vzri_lab/scratch/qlin/LFM-factorization/';
Input.output_name= ['SID_' Rawdata_name];



diary([Input.output_folder Input.output_name '.txt']) 

Input.rank=30;
Input.frames.start=311;
Input.frames.step=15; % Qian prefer bigger stepss
Input.frames.end=inf;
Input.step=2;                                                               
Input.prime=inf;
Input.bg_iter=2;
Input.rectify=1;
Input.Junk_size=2000;
Input.bg_sub=0;
Input.gpu_ids = [];
Input.num_iter=5;
Input.native_focal_plane=26;
Input.de_trend = 1;

Input.numworkers = 20;

output.Input=Input;
Input.update_template=1;


% Input.tmp_dir
% Input.step
% Input.step_
% Input.prime_
% Input.thres
% Input.nnmf_opts
% Input.recon_opts
% Input.update_template
% Input.fluoslide_fn


Input.recon_opts.lambda = [0, 0, 10];
Input.recon_opts.lambda_ = 0.1;
Input.recon_opts.p = 2;
Input.recon_opts.maxIter = 8;
Input.recon_opts.mode = 'TV';
Input.recon_opts.whichSolver = 'fast_nnls';

Input.thres = 10;
optional_args = struct;


% Input.frames.start = 1;
% Input.frames.steps = 10;
% Input.frames.end = 1e6;


crop_thresh_coord_x = 0.8;	%values for fish
crop_thresh_coord_y = 0.75;	%values for fish

%% start main_nnmf_SID from 
% ***********************Cache and open PSF***********************