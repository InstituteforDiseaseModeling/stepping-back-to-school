# "Stepping Back to School" report repository

This repository contains the code originally used for [IDM's](https://covid.idmod.org) school reopening analysis presented in the [Stepping Back to School](https://covid.idmod.org/data/Stepping_Back_to_School.pdf) modeling report, which is based on data from King County, WA, but explores fundamental COVID-19 transmission relationships that are broadly applicable. The report used the agent-based model Covasim, which can be downloaded from [GitHub](https://github.com/InstituteforDiseaseModeling/covasim) and used for other COVID-19 disease modeling.

* Code to implement schools in Covasim can be found in `covasim_schools`.
* The controller is implemented in `covasim_controller`.
* Other utility functions for running school analyses can be found in `school_tools`.
* Scripts to conduct the analysis are in `scripts`.
* Unit and integration tests are in `tests`.

## Installation


### Requirements

Python 3.7 or 3.8 (64-bit). (Note: Python 2 is not supported, Python <=3.6 requires special installation options, and Python 3.9 is not supported.)


### Steps

1. If desired, create a virtual environment.

    - For example, using [conda](https://www.anaconda.com/products/individual):

      ```
      conda create -n covaschool python=3.8
      conda activate covaschool
      ```

2. Install [SynthPops](https://github.com/InstituteforDiseaseModeling/synthpops), a package to create synthetic populations, in a folder of your choice. Note that `pip` installation does not currently include required Seattle data files:

   ```
   git clone https://github.com/InstituteforDiseaseModeling/synthpops
   cd synthpops
   python setup.py develop
   ```

3. Install this package (which will also install [Covasim](https://covasim.org)):

   ```
   git clone https://github.com/InstituteforDiseaseModeling/stepping-back-to-school
   cd stepping-back-to-school
   python setup.py develop
   ```


## Usage

Scripts in the `scripts` folder produce the results presented in the report. Each script generates a different set of results; not all are used in the report. Each script has a brief description of what it does. For a quick example, first run `python create_pops.py`, then run `python run_debug.py --show` (the `--show` argument makes the plots appear when running non-interactively). This script should take a few minutes to run.

For more information, see the documentation in the individual files.
