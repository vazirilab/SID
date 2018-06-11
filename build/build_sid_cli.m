%%
addpath(genpath('../external'));
addpath('../src');
addpath('../');
mcc -R -nodisplay -v -m nnmf_sid_cli.m -d ../bin -o nnmf_sid_cli