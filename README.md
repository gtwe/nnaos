# NNAOS Experiment Runner

This repository contains code to reproduce the experiments from the paper:

“Approximation, Estimation and Optimization Errors for a Deep Neural Network”
Gerrit Welper, Benjamin Keene

## Requirements

- GNU Make >= 3.82
- Python (tested on 3.12)
- *Nice to have*: Access to HPC/HTC

## Contents

There are two primary folders in this repository:
`experiments` and `reports`.

### `experiments`

Each experiment has its own named directory within `experiments`.

An experiment consists of:

- A parameter generation script `makeparams.py`
    - This produces multiple `exp_foo.yaml` in `./params/`
- An experiment runner script `runner.py`
    - This takes one parameter file as argument, `runner.py params/exp_foo.yaml`
    - This produces one `exp_foo.yaml` in `./results/`
    - This should be ran in parallel on HPC/HTC
- A script `dataframe.py` to collate results into `./results/df.pkl`
- A script `plotter.py` to create plots and TeX that will be saved into `reports`

### `reports`

A report consists of the numerical results from an experiment.

Like the `experiments` folder, each experiment has its own named
directory within `reports`. Inside each of these named directories
is a folder `plots`, containing loss surfaces, and a folder `tex`
which contains minimal LaTeX code for the tables in the paper.

Within `tables` is the skeleton code for the tables themselves in the paper.
This LaTeX code references the .tex files in the `tex` folder of the four experiments
of dimension 3 (dim3dep2_kernel, dim3dep2_mse, dim3dep5_kernel, dim3dep5_mse).
Note, these experiment directories will only be created by running their
respective `plotter.py` script, which relies upon the full experiment being ran.

## Experiments and Example

There are five experiments containing the `makeparams.py` file used
to generate the plots in the paper.

| Experiment | Description |
|------------|-------------|
| dim3dep2_kernel    | Depth 2 FCNN, x in R^3, varies over width and number of samples, trained with kernel loss |
| dim3dep2_mse       | Depth 2 FCNN, x in R^3, varies over width and number of samples, trained with MSE loss |
| dim3dep5_kernel    | Depth 5 FCNN, x in R^3, varies over width and number of samples, trained with kernel loss |
| dim3dep5_mse       | Depth 5 FCNN, x in R^3, varies over width and number of samples, trained with MSE loss |
| dim7dep2_big_batch_mse | Depth 2 FCNN, x in R^7, varies over many different widths and over a large number of samples, trained with MSE loss |
| example | Example experiment, used to test the code and run on a single machine |

A Makefile is provided to run the 'example' experiment, by executing `make example`.

### HPC Suggestions

If you have access to HPC/HTC, you can run the experiments in parallel.

Manually run the `makeparams.py` script to generate the parameter files.

Index the parameter files in a Slurm job array, e.g.:
```bash
# ... usual slurm variables
#SBATCH --array=1-500
# ... activate python environment
YAML_FILES=(params/*.yaml)
YAML_FILE=${YAML_FILES[$SLURM_ARRAY_TASK_ID-1]}
python runner.py $YAML_FILE
```
and submit the Slurm script with `sbatch`.

After jobs have completed, manually run the `dataframe.py` and `plotter.py` scripts to collate results and create plots.

While each individual run takes less than 10 minutes, there are a total of 11,700 runs across the five experiments.
