# Seeded Iterative Demixing
Matlab implementation of the Seeded Iterative Demixing (SID) algorithm for extracting neuronal signals from Light Field Microscopy (LFM) recordings in scattering media, as described in these publications:

Nöbauer*, T., Skocek*, O., Pernía-Andrade, A. J., Weilguny, L., Martínez Traub, F., Molodtsov, M. I. & Vaziri, A.  
[*Video rate volumetric Ca2+ imaging across cortex using seeded iterative demixing (SID) microscopy.*](https://www.nature.com/articles/nmeth.4341)  
Nature Methods 14, 811–818 (2017).

Skocek*, O., Nöbauer*, T., Weilguny, L., Martínez Traub, F., Xia, C. N., Molodtsov, M. I., Grama, A., Yamagata, M., Aharoni, D., Cox, D. D., Golshani, P. & Vaziri, A.  
[*High-speed volumetric imaging of neuronal activity in freely moving rodents.*](https://www.nature.com/articles/s41592-018-0008-0)  
Nature Methods 15, 429–432 (2018).

## Documentation
- [Input argument reference](http://htmlpreview.github.com/?https://raw.githubusercontent.com/vazirilab/sid/master/docs/published_m_files/sid_config_manage.html)
- [Discussion of input arguments and general usage (Wiki)](https://github.com/vazirilab/sid/wiki/Description-and-usage)
- [Function documenation (generated from source code)](https://github.com/vazirilab/sid/tree/master/docs/published_m_files)

## System requirements
- Matlab Base
- Matlab Parallel computing toolbox
- Matlab Image processing toolbox
- Matlab Statistics and machine learning toolbox
- Matlab Curve fitting toolbox

Tested with Matlab R2017b on Red Hat Enterprise Linux 6.

A multi-core workstation with >32 GB is required.

One or more Tesla-grade GPUs with several GB of RAM are highly recommended for fast execution (see Matlab documentation for supported GPU models).

## Installation in Matlab
Clone/download and add the base folder and the `src/` and `external/` folders to Matlab path recursively.

## Binary builds
Binaries for command-line usage will be provided in the `bin/` directory (the current binaries are not yet functional).
