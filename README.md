
# Brain-Signals

## Online API

Under construction ...

## Setup

<details>
  <summary>How to train model on your data</summary>
<br/>

Please follow this guidelines to setup everything you need after you cloned the project.

### Create the virtualenv

```
pyenv virtualenv brain-signals && pyenv local brain-signals
```
### Environment Variables

Create a `.envrc` file the write `dotenv` in it

Create a `.env` file then add this variables :

`DATASETS_PATH` directory where the datasets are stored

`LOCAL_REGISTRY_PATH` directory where the models are stored

Example :

```
# absolute path required, no slash at the end
DATASETS_PATH=/home/<user>/code/<githubuser>/Brain-Signals/.data/processed_datasets
LOCAL_REGISTRY_PATH=/home/<user>/code/<githubuser>/Brain-Signals/.registry
```


### Direnv

Allow direnv with

```
direnv allow
```
### Gitignore

Create a `.gitignore` file, and add all the files you created.

Here is an example :

```
# System file and folders
.DS_Store
.python-version
__pycache__/

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
### You can now train a model

Check on main.py to understand how the program works

</details>

## Notion
Access the project's [Notion](https://www.notion.so/Brain-signal-61cb427d38804549a3bd49b269f6fb0b) to check the roadmap and bible.
