function [config_valid, config_out] = sid_config_manage(config_in)
%SID_CONFIG_MANAGER Summary of this function goes here
%   Detailed explanation goes here
if isunix()
    psf_cache_dir_default = '/dev/shm';
else
    psf_cache_dir_default = tempdir();
end

config_definition = {
    % first column is param name
    % second colum is validator function handle
    % third colum is default value; leave empty to indicate required value
    
    %%% REQUIRED PARAMS
    {'indir',                       @isfolder                              'required'};
    {'outdir',                      @is_in_existing_dir                    'required'};
    {'psffile',                     @is_existing_file                      'required'};
    {'x_offset',                    @is_double                             'required'};
    {'y_offset',                    @is_double                             'required'};
    {'dx',                          @is_double                             'required'};
    
    %%% OPTIONAL PARAMS
    
    %%% Frames to include
    {'frames.start',                @is_positive_integer_or_zero,          1};
    {'frames.step',                 @is_positive_integer_or_zero,          1};
    {'frames.end',                  @is_positive_integer_or_zero,          inf};
    % alternative: give explicit list
    {'frames.list',                 @isvector,                             []};
    % set to true to average over frames in between that ones specified by start:step:end or 
    {'frames.mean',                 @islogical,                            true};
    
    %%% Mask for valid pixels
    % image file, in which pixels that should be ignored  are set to 0, others to 1. Use this to ignore irrelevant areas in the input frames.
    {'mask_file',                   @is_existing_file,                     ''};
    
    %%% Physical parameters, neuron size
    % ratio between the physical length of a voxel in the axial direction vs. the physical length in the lateral direction
    {'axial',                       @is_double,                            4};
    % typical neuron radius in px. Typically 6 for fish using 20x/0.5NA objective, 9-12 for mouse cortex and 16x/0.8NA
    {'neur_rad',                    @is_double,                            8};
    %
    {'native_focal_plane',          @isinteger,                            26};
    
    %%% LFM frame rectification
    {'rectify',                     @islogical,                            true};
    
    %%% Other global options
    % output filename prefix for all generated files
    {'SID_output_name',             @isstring,                             ['sid_result_' datestr(now, 'YY-mm-ddTHHMM') '.mat']};
    %
    {'psf_cache_dir',               @is_in_existing_dir,                   psf_cache_dir_default};
    %
    {'tmp_dir',                     @is_in_existing_dir,                   tempdir()};
    % default: []. List of GPU IDs to use throughout SID (Note that in Matlab, gpu_ids start with 1, not 0, as in the output of nvidia-smi)
    {'gpu_ids',                     @isvector,                             []};
    % use standard deviation instead of 2-norm of residual in merit functions of all optimizers
    {'use_std',                     @islogical,                            false};
    
    %%% Background subtraction
    {'bg_iter',                     @isinteger,                            2};
    {'bg_sub',                      @islogical,                            true};

    %%% Detrending
    % boolean, whether to perform detrending prior to NNMF
    {'detrend',                     @islogical,                            true};
    % integer, half width of sliding window (in units of frames) for low-pass filtering the frame means prior to detrending. Set this to a value that is largecompared to the duration of a Ca transient (e.g. 10 times as large), to avoid that the detrending smoothes out true Ca transients.
    {'delta',                       @isinteger,                            100};

    %%% Cropping
    % number of microlenses to crop from input frames on each side [left right top bottom], to avoid border artefacts
    % When giving a value of floor([ix1_lo_border_width ix1_hi_border_width ix2_hi_border_width ix2_hi_border_width] / Nnum)
    % that means that
    % cropped_img = full_img(ix1_lo_border_width + 1 : end - ix1_hi_border_width, ix2_lo_border_width + 1 : end - ix2_hi_border_width)
    {'crop_border_microlenses',     @isvector,                             [0 0 0 0]};
    %
    {'do_crop',                     @islogical,                            true};
    % two-element vector. If not given or empty, user is asked for interactive input
    {'crop_params',                 @isvector,                             []};
    %
    {'crop_mask',                   @ismatrix,                             true};

    %%% Low-rank NNMF
    % 
    {'nnmf_opts.rank',              @isinteger,                            30};
    %
    {'nnmf_opts.max_iter',          @isinteger,                            600};
    %
    {'nnmf_opts.ini_method',        @isstring,                            'pca'};
    %
    {'nnmf_opts.lamb_spat',         @isfloat,                              0};
    %
    {'nnmf_opts.lamb_temp',         @isfloat,                              0};
    %
    {'nnmf_opts.lamb_corr',         @isfloat,                              0};
    %
    {'nnmf_opts.lamb_orth_L1',      @isfloat,                              5e-4};
    %
    {'nnmf_opts.lamb_orth_L2',      @isfloat,                              0};
    %
    {'nnmf_opts.lamb_spat_TV',      @isfloat,                              0};
    %
    {'nnmf_opts.lamb_temp_TV',      @isfloat,                              0};
    % enable xval
    {'nnmf_opts.xval_enable',       @islogical,                            false};
    % number of partitions in which the data is decomposed
    {'nnmf_opts.xval_num_part',     @isinteger,                            5};
    % paramter range of the multiplier that needs to be scanned by xval. For default, see xval.m
    {'nnmf_opts.xval_param',        @islogical,                            []};

    %%% LFM reconstruction
    % TODO: make sure this actually gets used
    {'recon_opts.maxIter',          @isinteger,                            8};
    %
    {'recon_opts.lamb_L1',          @isfloat,                              0.1};
    %
    {'recon_opts.lamb_L2',          @isfloat,                              0};
    %
    {'recon_opts.lamb_TV_L2',       @isfloat,                              0};
    %
    {'recon_opts.ker_shape',        @isstring,                             'user'};
    %
    {'optimize_kernel',             @islogical,                            true};
    % band-pass filter reconstructed volume
    {'filter',                      @islogical,                            false};    

    %%% Segmentation
    %
    {'segmentation.threshold',      @isfloat,                              0.01};
    %
    {'segmentation.top_cutoff',     @isinteger,                            1};
    % if left [], defaults to size(psf_ballistic.H,5)
    {'segmentation.bottom_cutoff',  @isinteger,                            []};
    %
    {'cluster_iter',                @isinteger,                            40};

    %%% Neuron footprint dictionary generation
    %
    {'use_std_GLL',                 @islogical,                            false};

    %%% Template generation
    %
    {'template_threshold',          @isfloat,                              0.01};

    %%% Bi-convex optimization
    % number of iterations, where one iteration consist of a spatial and a temporal update
    {'num_iter',                    @isinteger,                            4};
    % used in reg_nnls.m
    {'SID_optimization_args.spatial_lamb_L1', @isfloat,                    0};
    % used in reg_nnls.m
    {'SID_optimization_args.spatial_lamb_L2', @isfloat,                    0};
    % used in reg_nnls.m
    {'SID_optimization_args.spatial_lamb_orth_L1', @isfloat,               1e-4};
    % used in LS_nnls.m
    {'SID_optimization_args.temporal_lambda', @isfloat,                    1e-4};
    %
    {'update_template',                       @islogical,                  true};
    
    %%% Timeseries extraction
    % unclear purpose, used in incremental_temporal_update_gpu()
    {'ts_extract_chunk_size',       @isinteger,                            200};
    
    %%% Reconstruct final (SID-demixed) neuron spatial filters
    %
    {'recon_final_spatial_filters', @islogical,                            false};
};


%% Loop over fields in config_definition. Check if it exists in config_in. If yes, use that value. Else, use default
% TODO: support two input modes: varargin, and a struct
% TODO: how to handle sub-structs for varargin?
% option 1) convert them to structs explicitly wherever they are used
% option 2) convert them to structs explicitly here
% option 3) convert them to structs automatically here: detect dots in arg names, and interpret (up to two levels!)
config_out = struct;
config_valid = true;
for i = 1:size(config_definition, 1)
    field_name = config_definition{i}{1};
    substruct_dot_index = strfind(field_name, '.');
    validator = config_definition{i}{2};
    default = config_definition{i}{3};
    
    if isempty(substruct_dot_index)
        if isfield(config_in, field_name) && validator(config_in.(field_name))
            config_out.(field_name) = config_in.(field_name);
        elseif ~strcmp(default, 'required')
            config_out.(field_name) = default;
        else
            config_valid = false;
            disp(['Required argument not given:' field_name]);
        end
    else
        substruct_name = field_name(1:substruct_dot_index-1);
        substruct_field_name = field_name(substruct_dot_index + 1 : end);
        if isfield(config_in, substruct_name) && isfield(config_in.(substruct_name), substruct_field_name) && validator(config_in.(substruct_name).(substruct_field_name))               
            config_out.(substruct_name).(substruct_field_name) = config_in.(substruct_name).(substruct_field_name);
        else 
            config_out.(substruct_name).(substruct_field_name) = default;
        end
        if strcmp(default, 'required')
            error('Required arguments cannot be sub-structs. Make it a top-level argument.');
        end
    end
end

end
