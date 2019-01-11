function [config_valid, config_out] = sid_config_manage(varargin)
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
    % third colum is default value; left empty to indicate required value

    
%%% REQUIRED PARAMS

    % Input directory containing raw LFM frames in single-page tiff files
    {'indir',                       @isfolder                              'required'                       @strip};
    
    % Directory to place output files in
    {'outdir',                      @is_in_existing_dir                    'required'                       @strip};
    
    % LFM PSF file as generated using LFrecon package 
    % (https://media.nature.com/original/nature-assets/nmeth/journal/v11/n7/extref/nmeth.2964-S2.zip)
    {'psffile',                     @is_existing_file                      'required'                       @strip};
    
    % Horizontal position of center of central microlens in LFM raw frames
    % (determine using http://graphics.stanford.edu/software/LFDisplay/)
    {'x_offset',                    @isfloat                               'required'                       @str2num};
    
    % Vertical position of center of central microlens in LFM raw frames
    % (determine using http://graphics.stanford.edu/software/LFDisplay/)
    {'y_offset',                    @isfloat                               'required'                       @str2num};
    
    % Microlens pitch in LFM raw frames
    % (determine using http://graphics.stanford.edu/software/LFDisplay/)
    {'dx',                          @isfloat                               'required'                       @str2num};
    

    
%%% OPTIONAL PARAMS
    
 %%% Frames to include
    
    % start:step:end indices of frames to use for demixing
    {'frames.start',                @is_positive_integer_or_zero,          1};
    {'frames.step',                 @is_positive_integer_or_zero,          1};
    {'frames.end',                  @is_positive_integer_or_zero,          inf};
    
    % Alternative: give explicit list
    {'frames.list',                 @(x) isnumeric(x) && all(floor(x) == x), []};
    
    % Set to true to average over frames in between that ones specified by 
    % start:step:end
    {'frames.mean',                 @islogical,                            true};
    
 %%% Mask for valid pixels
    
    % Image file, in which pixels that should be ignored  are set to 0, 
    % others to 1. Use this to ignore irrelevant areas in the input frames.
    {'mask_file',                   @is_existing_file,                     ''};
    
 %%% Physical parameters, neuron size
    
    % Ratio between the physical length of a voxel in the axial direction vs. 
    % the physical length in the lateral direction
    {'axial',                       @isfloat,                              4};
    
    % Typical neuron radius in px. Typically 6 for fish using 20x/0.5NA 
    % objective, 9-12 for mouse cortex and 16x/0.8NA
    {'neur_rad',                    @isfloat,                              8};
    
    % Index of native focal plane (=resolution breakdown plane) in
    % reconstructed LFM stack
    {'native_focal_plane',          @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            26};
    
 %%% LFM frame rectification
 
    % Whether or not to rectify the input frames. Set to false if input
    % frames are pre-rectified.
    {'rectify',                     @islogical,                            true};
    
 %%% Other global options
 
    % Output filename prefix for all generated files
    {'SID_output_name',             @isstring,                             ['sid_result_' datestr(now, 'YY-mm-ddTHHMM') '.mat']};
 
    % Directory with very high access speed for PSF file caching
    % (ideally, use a RAM-disk such as /dev/shm)
    {'psf_cache_dir',               @is_in_existing_dir,                   psf_cache_dir_default};
    
    % Directory for temporary files
    {'tmp_dir',                     @is_in_existing_dir,                   tempdir()};
    
    % List of GPU IDs to use throughout SID. default: [].  
    % (Note that in Matlab, gpu_ids start with 1, not 0, as in the output of nvidia-smi)
    {'gpu_ids',                     @(x) isnumeric(x) && all(floor(x) == x), []};
    
    % Whether or not to use standard deviation instead of 2-norm of 
    % residual in merit functions of all optimizers
    {'use_std',                     @islogical,                            false};
    
 %%% Background subtraction
 
    % Number of iterations for low-rank matrix factorization during
    % background subtraction
    {'bg_iter',                     @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            2};
    
    % Enable background subtraction
    {'bg_sub',                      @islogical,                            true};

 %%% Detrending
 
    % Whether to perform detrending prior to NNMF
    {'detrend',                     @islogical,                            true};
    
    % Half width of sliding window (in units of frames) for low-pass filtering 
    % the frame means prior to detrending. Set this to a value that is large 
    % compared to the duration of a Ca transient (e.g. 10 times as large), 
    % to avoid that the detrending smoothes out true Ca transients.
    {'delta',                       @(x) isnumeric(x) && isscalar(x) && floor(x) == x,     100};

 %%% Cropping
 
    % Number of microlenses to crop from input frames on each side 
    % [left right top bottom] (as displayed in Fiji), to avoid border artefacts
    % When giving a value of 
    % floor([ix1_lo_border_width ix1_hi_border_width ix2_hi_border_width ix2_hi_border_width] / Nnum)
    % that means that
    % cropped_img = full_img(ix1_lo_border_width + 1 : end - ix1_hi_border_width, ix2_lo_border_width + 1 : end - ix2_hi_border_width)
    {'crop_border_microlenses',     @(x) isnumeric(x) && all(size(x) == [1 4]) && all(floor(x) == x),      [0 0 0 0]};
    
    % Whether to crop frames based on threasholds on standard deviation and
    % background
    {'do_crop',                     @islogical,                            true};
    
    % Two-element vector, first element is threshold on standard deviation 
    % used to crop frames. Second is cut-off threshold on background 
    % between microlenses.
    % NOTE: If not given or empty, user is asked for interactive input
    {'crop_params',                 @(x) isnumeric(x) && all(size(x) == [1 2]),                             []};
    
    % User-defined crop mask, given as an array of the same size as the
    % input frames
    {'crop_mask',                   @ismatrix,                             true};

 %%% Low-rank NMF 
 % for more detailed docs on all options, see fast_NMF.m and
 % https://github.com/vazirilab/sid/wiki/Description-and-usage#low-rank-nnmf
 
    % NMF rank
    {'nnmf_opts.rank',              @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            30};
    
    % Max. number of iterations to perform when computing the NMF
    {'nnmf_opts.max_iter',          @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            600};
    
    % Initialization for NMF. For options, see fast_NMF.m
    {'nnmf_opts.ini_method',        @isstring,                            'pca'};
    
    % Lagrange multiplier for spatial sparsity (L1 norm of spatial component matrix S)
    {'nnmf_opts.lamb_spat',         @isfloat,                              0};
    
    % Lagrange multiplier for temporal sparsity (L1 norm of temporal component matrix T)
    {'nnmf_opts.lamb_temp',         @isfloat,                              0};
    
    % Lagrange multiplier for temporal correlation (L2 norm of covariance matrix)
    {'nnmf_opts.lamb_corr',         @isfloat,                              0};
    
    % Lagrange multiplier for L1 norm of Gramian of S
    {'nnmf_opts.lamb_orth_L1',      @isfloat,                              5e-4};
    
    % Lagrange multiplier for L2 norm of Gramian of S
    {'nnmf_opts.lamb_orth_L2',      @isfloat,                              0};
    
    % Lagrange multiplier for L2 norm of Total Variation of S
    {'nnmf_opts.lamb_spat_TV',      @isfloat,                              0};
    
    % Lagrange multiplier for L2 norm of Total Variation of T
    {'nnmf_opts.lamb_temp_TV',      @isfloat,                              0};
    
    % Enable cross-validation
    {'nnmf_opts.xval_enable',       @islogical,                            false};
    
    % Number of partitions in which the data is decomposed for cross-validation
    {'nnmf_opts.xval_num_part',     @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            5};
    
    % Paramter range of the multiplier that needs to be scanned by xval. 
    % For default, see xval.m
    {'nnmf_opts.xval_param',        @islogical,                            []};

 %%% LFM reconstruction
 % for more detailed docs on all options, see reconstruct_S.m and
 % https://github.com/vazirilab/sid/wiki/Description-and-usage#lfm-reconstruction
 
    % Number of iterations of Richardson-Lucy updates for LFM deconvolution
    {'recon_opts.maxIter',          @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            8};
    
    % Lagrange multiplier for L1-norm of reconstructed LFM volume
    {'recon_opts.lamb_L1',          @isfloat,                              0.1};
    
    % Lagrange multiplier for L2-norm of reconstructed LFM volume
    {'recon_opts.lamb_L2',          @isfloat,                              0};
    
    % Lagrange multiplier for Total Variation of reconstructed LFM volume
    {'recon_opts.lamb_TV_L2',       @isfloat,                              0};
    
    % Neuron shape to allow during reconstruction
    {'recon_opts.ker_shape',        @isstring,                             'user'};
    
    % Enable automatic learning of neuron shape kernel my iteratively comparing
    % a convolution of current kernel shape with neuron centroids of reconstructed volume with
    % actual reconstructed volume
    {'optimize_kernel',             @islogical,                            true};
    
    % Band-pass filter the reconstructed NMF componet volumes. Warning: this can take tens of minutes per component and GPU
    {'filter',                      @islogical,                            false};    

 %%% Segmentation
 
    % Threshold for accepting local maxima as neuron candidates 
    % (increase to reduce over-segmentation)
    {'segmentation.threshold',      @isfloat,                              0.01};
    
    % Smallest z-plane index from which to accept segmentation centers as neuron condidates
    % (increase to exclude artefacts at top of volume)
    {'segmentation.top_cutoff',     @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            1};
    
    % Largest z-plane index from which to accept segmentation centers as neuron condidates
    % (decrease to exclude artefacts at bottom of volume)
    % if left [], defaults to size(psf_ballistic.H,5)
    {'segmentation.bottom_cutoff',  @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            []};
    
    % Number of iterations to perform when attempting to merge closely spaced 
    % neuron candidates from different NMF components by spatial clustering. 
    % Maximal size of clusters is determined by parameter neur_rad.
    {'cluster_iter',                @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            40};

 %%% Neuron footprint dictionary generation
 
    % If GPU support is enabled and this flag is set to true, generate_LFM_library_CPU.m 
    % is used for neuron footprint dictionary generation. Otherwise, generate_LFM_library_GPU.m 
    % is used. If GPU support is disabled, generate_LFM_library_CPU.m is used regardless of this flag.
    % (see https://github.com/vazirilab/sid/wiki/Description-and-usage#lfm-footprint-dictionary-generation)
    {'use_std_GLL',                 @islogical,                            false};

 %%% Template generation
 
    % Threshold that determines size of template radius for each neuron candidate.
    % Increase to generate larger templates
    {'template_threshold',          @isfloat,                              0.01};

 %%% Bi-convex optimization (main SID demixing)
 
    % Number of bi-convex SID demixing iterations, 
    % where one iteration consist of a spatial and a temporal update
    {'num_iter',                    @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            4};
    
    % Lagrange multiplier for L1-norm (sparsity) regularizer of spatial components during SID demixing
    {'SID_optimization_args.spatial_lamb_L1', @isfloat,                    0};
    
    % Lagrange multiplier for L2-norm regularizer of spatial components during SID demixing
    {'SID_optimization_args.spatial_lamb_L2', @isfloat,                    0};
    
    % Lagrange multiplier for L1-norm of Gramian matrix of spatial components during SID demixing
    {'SID_optimization_args.spatial_lamb_orth_L1', @isfloat,               1e-4};
    
    % Lagrange multiplier for L1-norm (sparsity) regularizer of temporal components during SID demixing
    {'SID_optimization_args.temporal_lambda', @isfloat,                    1e-4};
    
    % Increase size of templates after each iteration (enable to merge components and thus avoid false positive neurons)
    {'update_template',                       @islogical,                  true};
    
 %%% Timeseries extraction
 
    % Number of frames to extract timeseries from in one chunk (see incremental_temporal_update_gpu.m)
    % Decrease to avoid out-of-memory errors
    {'ts_extract_chunk_size',       @(x) isnumeric(x) && isscalar(x) && floor(x) == x,                            200};
    
 %%% Reconstruct final (SID-demixed) neuron spatial filters
 
    % Whether to reconstruct final (SID-demixed) neuron spatial filters
    % (entirely optional; not required for time series extraction)s
    {'recon_final_spatial_filters', @islogical,                            false};
};

%% Check varargin and assemble struct from key-value pars if necessary
if nargin == 1 
    if isa(varargin{1}, 'struct')
        config_in = varargin{1};
    else
        error('If only a one argument is given, it has to be a struct, with fields according to the required and optional parameters (see docs).');
    end
elseif nargin < 1 || mod(nargin, 2)
    error('Expecting either a single struct or a sequence of key-value pairs as input arguments.')
else
    % If the input argument is not a struct, then we expect a variable number of key-value pairs. 
    % The keys have to be strings. The values may be strings that can be parsed to the required data type
    config_in = struct;
    for i = 1:2:nargin
        key = varargin{i};
        val = varargin{i + 1};
        substruct_dot_index = strfind(key, '.');
        % find key in config_definition
        if isempty(substruct_dot_index)
            config_in.(key) = val;
        else
            substruct_name = key(1:substruct_dot_index-1);
            substruct_field_name = key(substruct_dot_index + 1 : end);
            config_in.(substruct_name).(substruct_field_name) = val;
        end
    end
end

%% Loop over fields in config_definition. Check if it exists in config_in. If yes, use that value. Else, use default
% TODO: add support for varargin input instead of struct
% TODO: add support for key-value pairs of strings as input (for compiled command line use)
config_out = struct;
config_valid = true;
for i = 1:size(config_definition, 1)
    field_name = config_definition{i}{1};
    substruct_dot_index = strfind(field_name, '.');
    validator = config_definition{i}{2};
    default = config_definition{i}{3};
    if numel(config_definition{i}) > 3
        parser = config_definition{i}{4};
    elseif isnumeric(default) || islogical(default)
        parser = @str2num;
    else
        parser = @(x) x;  % no operation
    end
    
    try
        if isempty(substruct_dot_index)
            if isfield(config_in, field_name)
                val = config_in.(field_name);            
                if validator(val)
                    config_out.(field_name) = val;
                elseif validator(parser(val))
                    config_out.(field_name) = parser(val);
                else
                    error(['Parse/validation error for key ' field_name '. Invalid value was : ' val]);
                end
            elseif ~strcmp(default, 'required')
                config_out.(field_name) = default;
            else
                config_valid = false;
                disp(['Required argument not given:' field_name]);
            end
        else
            substruct_name = field_name(1:substruct_dot_index-1);
            substruct_field_name = field_name(substruct_dot_index + 1 : end);
            if isfield(config_in, substruct_name) && isfield(config_in.(substruct_name), substruct_field_name)
                val = config_in.(substruct_name).(substruct_field_name);
                if validator(val)           
                    config_out.(substruct_name).(substruct_field_name) = val;
                elseif validator(parser(val))
                    config_out.(substruct_name).(substruct_field_name) = parser(val);
                else
                    error(['Parse/validation error for key ' field_name '. Invalid value was : ' val]);
                end
            else 
                config_out.(substruct_name).(substruct_field_name) = default;
            end
            if strcmp(default, 'required')
                error('Required arguments cannot be sub-structs. Make it a top-level argument.');
            end
        end
    catch my_error
        disp(['Error while processing input: key=' field_name ' val=' num2str(val) ':']);
        rethrow(my_error);        
    end
end

end
