# Kernel Learning - Deliverables

## Contents

The distributed package contains to folders:
- export: Here, you can find a static HTML export of the notebook. Please refer to this file if you have any trouble with the .ipynb within jupyter lab.
- src: Here, you can find the .ipynb file with should be opened within the docker container using jupyter lab. 

## Installation
Running the notebook with docker:
- Extract the "dist" folder and open a terminal from this folder 
- Make sure that the docker daemon is running on your system
- Write 'docker-compose up' to create container from image
- Copy-paste IP (e.g. http://127.0.0.1:1234/lab) into your browser
- Password for JupyterLab: 'learning'

The container size is about 1 GB after installation. Build time varies by downloading speeds.

Running the notebook outside of docker:
We recommend  to use this notebook in the provided docker container. Otherwise, you can face compatibly issues. At least run the notebook in a jupyter environment  and ensure, that the python requirements are met.

## Use
Please run the notebook from start to finish and avoid cross running of cells. Keep also in mind, that some cells are executing computations, which may take a while to finish on your system. 