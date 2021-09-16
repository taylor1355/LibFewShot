# Getting started

This section shows an example of a process using `LibFewShot`.

## Prepare dataset(use miniImageNet as example)

1. download and extract [miniimagent--ravi](https://drive.google.com/file/d/1Oq7JKbd8-6QgLXbZ1MW4Wkv39EgDBk5t/view?usp=sharing).

2. check the structure of the dataset：

    The dataset must be in following structure:

    ```
    dataset_folder/
    ├── images/
    │   ├── images_1.jpg
    │   ├── ...
    │   └── images_n.jpg
    ├── train.csv *
    ├── test.csv *
    └── val.csv *
    ```

## Modify config file

Use`ProtoNet` as example：
1. create a new `yaml` file `getting_started.yaml` in `config/`
2. write the following to the created file:
   ```yaml
   includes:
     - headers/data.yaml
     - headers/device.yaml
     - headers/losses.yaml
     - headers/misc.yaml
     - headers/model.yaml
     - headers/optimizer.yaml
     - classifiers/Proto.yaml
     - backbones/Conv64F.yaml
   ```

More details refer to  [write a config yaml](./tutorials/t0-write_a_config_yaml.md).

## Run

1. modify `run_trainer.py` as follows:
    ```python
    # -*- coding: utf-8 -*-
    import sys

    sys.dont_write_bytecode = True

    from core.config import Config
    from core import Trainer

    if __name__ == "__main__":
        config = Config("./config/getting_started.yaml").get_config_dict()
        trainer = Trainer(config)
        trainer.train_loop()
    ```
2. train with console command:
   ```shell
   python run_trainer.py
   ```
3. wait for the end of training.

## View log files

After running the program, you can find a symlink `results/ProtoNet-miniImageNet-Conv64F-5-1` and a directory`results/ProtoNet-miniImageNet-Conv64F-5-1-$TS` that `TS` means timestamp, which contains two directorys: `checkpoint/` and `log_files/`, and a configuration file:`config.yaml`. Note that the symlink will always link to the directory createdthe last time you trained with the same few shot learning configuration.

`config.yaml` contains all the settings used in the training.

`log_files/` contains tensorboard files, training log files and test log files.

`checkpoints/` contains model checkpoints saved at `$save_insterval` intervals, last model checkpoint(used to resume) and best model checkpoint(used to test). The checkpoint files are generally divided into `emb_func.pth`, `classifier.pth`, and `model.pth` , a combination of the first two.