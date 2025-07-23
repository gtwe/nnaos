import sys
import os

sys.path.append('../../')

import aos.plots as plots
import aos.tables as tables

if __name__ == "__main__":
    expname = os.path.basename(os.getcwd())
    plots.exp_plotter(expname)
    tables.make_tex_files(expname)
