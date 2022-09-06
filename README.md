# Brain-Signals

## Setup the project

Please follow this guidelines to setup everything you need.
### Create the virtualenv

```bash
  pyenv virtualenv brain-signals && pyenv local brain-signals
```
### Environment Variables

Create a *.env* file then add this variables :

`DATASETS_PATH` directory where the datasets are stored

`LOCAL_REGISTRY_PATH` directory where the models are stored


### Direnv

Allow direnv with

```bash
  direnv allow
```
### Gitignore

Create a *.gitignore* file, and add all the files you created.

Here is an example :
```
# System file and folders
.DS_Store
.python-version

# Personnal things
.Jupyter_notebooks/
.env
.gitignore

# Too heavy for gitignore
.data
.registry

# Executables
*.egg-info
```
## You can now train a model

Check on main.py to understand how the program works
