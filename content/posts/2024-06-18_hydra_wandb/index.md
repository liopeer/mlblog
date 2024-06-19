+++
title = 'Hydra and WandB for Machine Learning Experiments'
author = 'Lionel Peer'
date = 2024-06-18T20:03:18+02:00
draft = false
+++
## Introduction
Hydra [^1] and WandB [^2] have become indispensable tools for me when tracking my machine learning experiments. In this post I would like to share how I combine these two tools for maximum reproducibility, debuggability and flexibility in experiment scheduling. This post is very much a personal knowledge resource for me, therefore I will try to keep it up-to-date when my workflow changes. I want to cover the following things:

1. build a sensible config hierarchy that never requires you to change multiple files
2. using common project names and run names across WandB and Hydra
3. debug your code without excessive logging from WandB and Hydra

#### WandB
At the time of initially writing this post (June 2024) I have been using WandB for about a year and while its feature set is massive, I use it almost exclusively for logging during training, thinking of it mostly as a **tensorboard on steroids**. Especially the automatic logging of the hardware useage has significantly improved my ability to squeeze every last FLOP out of my hardware.

#### Hydra
Is a tool from facebook research that builds on top of OmegaConf [^4] and is specifically meant for ML experiment launching and tracking:

1. let's you recursively build `yaml`-based hierarchical configurations for maximum flexibility and quick exchange of components without rewriting the configs
2. includes a CLI (command-line interface) launcher supporting overrides of any configs
3. creates an experiment folder for every experiment you launch through the CLI

Hydra is extremely powerful, but unfortunately that also means that it takes a bit of time to get comfortable with it - maybe a bit too much for that it only handles your configs. But since I learned it anyways, let me give you an easy introduction.

### Hydra Preliminaries
In many applications we use json or yaml files to store our configurations, so that they are easily accessible. In machine learning it is further often the case that we want to be able to quickly exchange certain parts of our pipeline: Maybe we are working with several datasets and we would like to be able to switch between them. Or we want to do a simple change like tweaking the learning rate. In a naive setting you would create a new config file for each case, leading not only to a cluttered config directory, but also to larger problems once you change something in your code - and you'll have to change ALL of those files. Hydra solves this for you by allowing for a very high abstraction in your configs, following the [DRY - don't repeat yourself](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principle, so that when your code changes you also only have to change one file in your configs.

#### The Ultimate Basics
Hydra provides the decorator `@hydra.main`, where you will specify `config_path`, which will point to the directory in your repo that contains the configuration files. Assuming a repo structure like this
```txt
myrepo
├── config
│   ├── my_config.yaml
│   └── my_other_config.yaml
└── main.py
```
a minimal `main.py` would look like
```python
# main.py
import hydra

@hydra.main(version_base=None, config_path="config", config_name="my_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

if __name__ == "__main__":
    main()
```
where `config_name` is the default config that is being chosen. Launching your script with
```bash
python main.py
```
will pass the config to the main function and the default config can be overridden by launching with:
```bash
python main --config-name=my_other_config
```
If you don't specify a default config, then of course the flag `--config-name` is not optional but required. The config object is similar to a dictionary, but its keys can be accessed by the dot-notation – like class attributes.

I encourage you to play around with two different yaml files a bit before continuing. As you might realize, Hydra creates a new directory in `./output` every time you run your command. This directory is supposed to hold all your logs and we will make use of that later on.

#### Overrides
Assuming your `my_config.yaml` contains keys
```yaml
lr: 0.001
batch_size: 16
model:
  hidden_layers: 5
  in_channels: 3
```
then Hydra allows you to easily override those values by calling :
```bash
python main --config-name=my_config model.hidden_layers=4 lr=0.01
```
You can also add keys that did not exist yet
```bash
python main --config-name=my_config +device=gpu
```
or you can enforce that a certain value is passed on the command line by setting
```yaml
lr: 0.001
batch_size: 16
model:
  hidden_layers: 5
  in_channels: ???
```
which will throw an error if the flag `model.in_channels=<VALUE>` is not passed.
#### Defaults
As mentioned before, the strength of Hydra is in the ability to hierarchically structure your configs and combine them together. Combining different yaml files into a single config is easy with the `defaults` directive: Assume we now add a folder `model` with different model configurations
```txt
myrepo
├── config
│   ├── globals.yaml
│   ├── model
│   │   ├── alexnet.yaml
│   │   └── resnet.yaml
│   ├── my_config.yaml
│   └── my_other_config.yaml
└── main.py
```
and the `resnet` containing the hyperparameters
```yaml
hidden_layers: 12
input_channels: 3
```
then adding the defaults directive to our configuration
```yaml
# my_config.yaml
defaults:
  - model: resnet
lr: 0.001
batch_size: 16
```
would at runtime yield a structure where the `resnet` hyperparameters are accessible under the key `model`:
```yaml
model: 
  hidden_layers: 12
  input_channels: 3
lr: 0.001
batch_size: 16
```
Simple, right? Hydra calls the different `model` a **group**.

Apart from importing groups (which is a hiearchical procedure), you can also merge several configs from the same hierarchy together using the `defaults` directive: Some global variables like paths might be shared by `my_config` and `my_other_config`. You can import them by additionally calling:
```yaml
defaults:
  - model: alexnet
  - globals
lr: 0.001
batch_size: 16
```
{{< notice note >}}
Usually you're supposed to include `_self_` – a reference to the local config – in the defaults as well. Putting it before or after the other defaults defines the [resolution order](https://hydra.cc/docs/advanced/defaults_list/) if keys appear several times.
{{< /notice >}}

#### Resolvers
As hydra is built on top of OmegaConf, you can use [OmegaConf resolvers](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html), which are functions inside of your yaml file that let you do operations on the hyperparameters, e.g. if you have a parameter `devices: [0,1]` or `devices: 1` in your yaml, you could create a resolver `isdist` that sets a different parameter `distributed` as either `true` or `false`, depending on `devices`. I tend to define such functions in a separate file `custom_resolvers.py`:
```txt
myrepo
├── config
│   ├── globals.yaml
│   ├── model
│   │   ├── alexnet.yaml
│   │   └── resnet.yaml
│   ├── my_config.yaml
│   └── my_other_config.yaml
├── custom_resolvers.py
└── main.py
```

```python
# custom_resolvers.py
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("isdist", lambda x: len(x)>1)
```
and making sure they are imported into your main file:
```python
import custom_resolvers
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
```
Once that is done, you can call the resolver in your yaml as:
```yaml
devices: [0,1]
distributed: ${isdist:${devices}}
```
As you can see, resolvers and variables are wrapped in `${}` and you can also create resolvers that can also also take multiple arguments `${resolvername:${arg1},${arg2}}`.

## Launching Actual Experiments
We will now apply the acquire knowledge to an actual project where we want to compare the performance of SFNOs (spherical Fourier neural operator) [^5] and FNO (Fourier neural operator) [^6] for solving a PDE (partial differential equation) called the *Shallow Water Equations*. It is not important to understand the models or the PDE, our model should simply learn to map one image to a different one.

All the code is available at ([GitHub - SFNO-ShallowWater](https://github.com/liopeer/sfno_shallowwater)) if you want to follow along.

### Setting up our `config` directory
I choose to create the following groups in my `config`:
1. `training`: Training hyperparameters that influence my training result, such as the effective batch size, learning rate and the max number of epochs. Other training hyperparameters that don't influence my model performance – such as the hardware specification – are not included in here.
2. `data`: Dataset-related hyperparameters, such as image resolution and the train/val split.
3. `model`: Model hyperparameters, such as the number of hidden layers and their sizes.
4. `paths`: Includes absolute paths to e.g. datasets for different machines or for different users that are working on the project.

This creates a structure like this (see the github repo for the file contents):
```txt
config
├── data
│   ├── 32x64.yaml
│   ├── 64x128.yaml
│   └── data_globals.yaml
├── globals.yaml
├── model
│   ├── fno.yaml
│   ├── model_globals.yaml
│   └── sfno.yaml
├── neuraloperator.yaml
├── paths
│   ├── cluster.yaml
│   └── home_desktop.yaml
└── training
    └── train_default.yaml
```
As you can see, for the `data` I created two different files for different resolutions, together with a file `data_globals` that includes shared hyperparameters between the two. The same goes for the `model`, where we also have hyperparameters for two different models and a number of shared hyperparameters in a global file. For training I currently stick with a single config and on the top level I split between `globals` where I define logging hyperparameters that are unlikely to change and `neuraloperator`, where all the configs come together.

### Creating Log Directories for each Run
#### Customizing the Directory Scheme
Hydra creating a log directory for every run is great – until you want to debug your code and you end up with hundreds of run folders that are meaningless. We therefore want to conditonally want to reroute the output to a `output/debug` directory if we call our script in debug mode:
```bash
python train.py --config-name=neuraloperator debug=True
```
Let's therefore try to create a logging structure that sorts our runs by date/time, names them by the utilized model and image resolution and by any hyperparameters that were overridden.
```bash
outputs
├── 2024-06-19
│   ├── 2024-06-19_16-51-43_sfno64x128_data.num_examples=50_training.max_epochs=1
│   │   ├── ckpt
│   │   ├── train.log
│   │   └── wandb
│   └── 2024-06-19_17-10-49_sfno64x128_data.num_examples=50_training.max_epochs=1
│       ├── ckpt
│       ├── train.log
│       └── wandb
└── debug
    ├── ckpt
    └── train.log
```
The `debug` directory will simply be overwritten, keeping everything clean and the actual runs will be grouped by date.

Overriding the output directory scheme requires the setting
```yaml
hydra:
  run:
    dir: some_output_dir
```
in your config and we can use the `now` resolver (included in Hydra) to get time/date
```yaml
${now:%Y-%m-%d}
```
and Hydra also provides the `hydra` resolver that let's us access e.g. the model choice we made (`sfno` or `fno`)
```yaml
${hydra:runtime.choices.model}
```
and any override argument with
```yaml
${hydra:job.override_dirname}
```
with these tools we have everything we need to build our custom directory naming scheme
```yaml
output_dir_scheme: ${now:%Y-%m-%d}/${now:%Y-%m-%d_%H-%M-%S}_${hydra:runtime.choices.model}${hydra:runtime.choices.data}_${hydra:job.override_dirname}
```
{{< notice info >}}
If you have any good way of wrapping this onto several lines in my yaml file please let me know 🙏😁. It's so ugly!
{{< /notice >}}
and we can include our scheme:
```yaml
hydra:
  run:
    dir: outputs/${output_dir_scheme}
```
If you run this with without any overrides, you will see that it leaves an ugly underscore at the end (when `job.override_dirname`) is empty. In my repository I therefore additionally wrap this in a custom resolver, but this is really just cosmetics at this point:
```python
OmegaConf.register_new_resolver("prepend_underscore", lambda x: "" if len(x)==0 else "_"+x)
```
```yaml
output_dir_scheme: ${now:%Y-%m-%d}/${now:%Y-%m-%d_%H-%M-%S}_${hydra:runtime.choices.model}${hydra:runtime.choices.data}${prepend_underscore:${hydra:job.override_dirname}}"
```
#### Rerouting the Output during Debugging
We now have nice run directories, but we still need to reroute the output in case the `debug=True` flag is set. I did this by defining another resolver to which I pass my directory scheme and the debug flag, which either returns the directory scheme or the debug directory:
```python
def output_dir(output_dir: str, debug: bool):
    assert isinstance(debug, bool), type(debug)
    if debug:
        return "debug"
    else:
        return output_dir

OmegaConf.register_new_resolver("output_dir", output_dir)
```

```yaml
hydra:
  run:
    dir: ${output_dir}

log_dir: ./outputs
output_dir_scheme: ${now:%Y-%m-%d}/${now:%Y-%m-%d_%H-%M-%S}_${hydra:runtime.choices.model}${hydra:runtime.choices.data}${prepend_underscore:${hydra:job.override_dirname}}"
output_dir: ${log_dir}/${output_dir:${output_dir_scheme},${debug}}
debug: False
```
### Logging Checkpoints
I am using PyTorch Lightning with the `ModelCheckpoint` callback, that takes the argument `dirpath` to specify where to save the checkpoints. I can now simply add to my config:
```yaml
ckpt_dir: ${mkdirs:${output_dir}/ckpt}
```
### Logging to WandB
In order to also save my WandB run in the same directory I need to pass the `save_dir` argument to Lightning's `WandbLogger`. In my experience this only works if you give it an absolute path, therefore I decided to add an `abspath` resolver
```python
OmegaConf.register_new_resolver("abspath", lambda x: os.path.abspath(x))
```
```yaml
wandb_dir: ${abspath:${output_dir}}
```
and in order to be able to reassociate my local folder with my online loggings I use the output scheme from before to name my runs:
```yaml
wandb_run_name: ${output_dir_scheme}
```
## Closing Words
If you made it to the end: Thank you very much! And if you have any helpful additions, please reach out via my socials. 🤗

[^1]: O. Yadan, Hydra - A framework for elegantly configuring complex applications. Github, 2019. [Online]. Available: https://github.com/facebookresearch/hydra
[^2]: Biewald, L. (2020). Experiment Tracking with Weights and Biases. https://www.wandb.com/ 
[^3]: Yoo, A.B., Jette, M.A., Grondona, M. (2003). SLURM: Simple Linux Utility for Resource Management. In: Feitelson, D., Rudolph, L., Schwiegelshohn, U. (eds) Job Scheduling Strategies for Parallel Processing. JSSPP 2003. Lecture Notes in Computer Science, vol 2862. Springer, Berlin, Heidelberg. https://doi.org/10.1007/10968987_3
[^4]: Yadan, O., Sommer-Simpson, J., & Delalleau, O. (2019). omegaconf [Computer software]. https://github.com/omry/omegaconf
[^5]: Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).
[^6]: Bonev, Boris, et al. "Spherical fourier neural operators: Learning stable dynamics on the sphere." International conference on machine learning. PMLR, 2023.
[^7]: Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) [Computer software]. https://doi.org/10.5281/zenodo.3828935