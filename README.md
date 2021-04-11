# Response Type Analysis
This repository contains my analysis of the Question Response Type problem, found [here](https://cogcomp.seas.upenn.edu/Data/QA/QC/). My code and analysis is found in the "Question Response Type Analysis.ipynb" file, and an html version of that file has been included for easy viewing. The utils.py file contains some utility plotting code (I hate putting that in notebooks, and it's easier to write in a real editor). The models I generated are stored in the *models* folder and can be loaded with joblib. Finally, we have two pieces that should make this reproducible - a devcontainer file that builds a docker container in VS Code that is the same environment I worked in, and a pip freeze if you don't want to work in a devcontainer. 