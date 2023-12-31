[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8424585.svg)](https://doi.org/10.5281/zenodo.8424585)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PTB-M4D/time-gating-examples/main?labpath=examples%2Finteractive_gating_with_unc.ipynb)

# Time Gating Examples

This repository holds examples for gating s-parameters in the time-domain.

Click the "launch binder" badge above to open the notebook showing the traditional time gating process.

## Recommendations for Local Execution

The provided script operates with large covariance matrices and is not really suited to be used in mybinder.org.

### Clone the Repository

```bash
git clone https://github.com/PTB-M4D/time-gating-examples.git
```

### Install Python

Install a fairly recent version of Python (3.9+) on your machine.

### Setup a new Python Environment

On a (Linux) shell or (Windows) powershell execute:

```bash
# create directory for your python environments
cd ~
mkdir python_envs

# create new environment for this project
cd ~/python_envs
python -m venv time_gating

# activate the new python environment
source ~/python_envs/time_gating/bin/activate
# or on powershell: 
# ~\python_venv\time_gating\Scripts\Activate.ps1

# change into project's git repo
cd ~/path/to/git/repo/time-gating-examples

# setup the environment by installing the requirements
pip install -r requirements.txt
```

### Suggestion for an IDE which supports Python + Jupyter

- VS Code:
  - <https://code.visualstudio.com/>
  - Extensions (installable from within VS Code):
    - Python: <https://marketplace.visualstudio.com/items?itemName=ms-python.python>
    - Jupyter: <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>
