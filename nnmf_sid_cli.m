function [ ret ] = nnmf_sid_cli(indir, outdir, psffile, offset_x, offset_y, dx, varargin)
%%
disp([datestr(now) ': Starting nnmf_sid_cli']);

parser = inputParser;
parser.addRequired('indir', @(f) is_existing_file_or_dir(f, {'tif' 'tiff'}));
parser.addRequired('outdir');
parser.addRequired('psffile', @(f) is_existing_file(f, 'mat'));
parser.addRequired('offset_x', @is_double);
parser.addRequired('offset_y', @is_double);
parser.addRequired('dx', @is_double);

parser.addParameter('step', 1, @is_positive_integer_or_zero);
parser.addParameter('step_', 3, @is_positive_integer_or_zero);
parser.addParameter('rank', 30, @is_positive_integer_or_zero);
parser.addParameter('rectify', 1, @is_positive_integer_or_zero);
parser.addParameter('bg_iter', 2, @is_positive_integer_or_zero);
parser.addParameter('junk_size', 1000, @is_positive_integer_or_zero);
parser.addParameter('bg_sub', 1, @is_positive_integer_or_zero);
parser.addParameter('prime', 40000, @is_positive_integer_or_zero);
parser.addParameter('prime_', 4800, @is_positive_integer_or_zero);
parser.addParameter('n_iter', 4, @is_positive_integer_or_zero);
parser.addParameter('native_focal_plane', 26, @is_positive_integer_or_zero);

parser.addParameter('testing', 0, @is_positive_integer_or_zero);
parser.addParameter('tmp_dir', tempdir(), @is_in_existing_dir);
parser.addParameter('out_filename', '');
parser.addParameter('out_param_file', 'input_and_derived_parameters.json');
parser.addParameter('gpu_ids', '[]');

parser.parse(indir, outdir, psffile, make_double(offset_x), make_double(offset_y), make_double(dx), varargin{:});

p = parser.Results;
for parameter = {'offset_x' 'offset_y' 'dx' 'step' 'step_' 'rank' 'rectify' 'bg_iter' ...
        'junk_size' 'bg_sub' 'prime' 'prime_' 'n_iter' 'native_focal_plane'}
    p.(parameter{:}) = make_double(parser.Results.(parameter{:}));
end

%% Create necessary directories
if ~exist(p.outdir, 'dir')
    disp(['Creating dir for output data: ' p.outdir]);
    mkdir(p.outdir);
end

[~, rand_string] = fileparts(tempname());
p.tmp_dir = fullfile(p.tmp_dir, ['nnmf_sid_' rand_string]);
disp(['Creating tmp dir: ' p.tmp_dir]);
mkdir(p.tmp_dir);

disp('All input and derived parameters: ');
disp(p);

%% Call nnmf_sid function
disp([datestr(now) ': Calling main_nnmf_SID()']);
if ~p.testing
    main_nnmf_SID(p.indir, p.outdir, p.psffile, p.offset_x, p.offset_y, p.dx, p);
end

%%
disp([datestr(now) ': main_nnmf_SID() complete.']);

disp([datestr(now) ': Writing output parameter file']);
savejson('input_and_derived_parameters', p, p.out_param_file);

if ~strcmp(p.tmp_dir, '')
    disp([datestr(now) ': Deleting job tmp dir']);
    rmdir(p.tmp_dir, 's');
end

ret = 0;
disp([datestr(now) ': Returning.']);
if isdeployed()
    exit(ret);
end
end
