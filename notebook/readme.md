# Overview

This notebook contains three main files for users to replicate Figure 1-panel a,b,c, and Figure 2.

FiguresCode.py, FiguresCode_test.ipynb, plot_maps.py are the source code for plot_fun.py, plot_run.py, and plot_trial.ipynb. 

plot_fun.py is a Python script consisting of three functions to generate plots for the PNAS manuscript using source code from FiguresCode.py and plot_maps.py. 

plot_run.py is a Python script using functions from plot_fun.py to generate the PNG output.

plot_trial.ipynb is a Jupyter Notebook providing a more flexible environment for readers to replicate the figures. 

# Installation

To successfully run the scripts on your computer, we recommend you ensure you have the most updated version of the following libraries: pandas, numpy, geopandas, matplotlib, colormap, cartopy, plotly, kailedo.

We propose an installation instruction with Anaconda3 (version 2.3.2) with the newest python version (tested at python 3.10.4)

- Install libraries using conda command: "conda install -c conda-forge numpy"
- Install libraries using conda command: "conda install -c conda-forge pandas"
- Install libraries using conda command: "conda install -c conda-forge matplotlib"
- Install libraries using conda command: "conda install -c conda-forge colormap"
- Install libraries using conda command: "conda install -c conda-forge cartopy"
- Install libraries using conda command: "conda install -c conda-forge plotly"
- Install libraries using conda command: "conda install -c conda-forge fiona"
- Install libraries using conda command: "conda install -c conda-forge seaborn"
- Install libraries using conda command: "conda install -c conda-forge geopandas"
- Install libraries using conda command: "conda install -c conda-forge python-kaleido"
