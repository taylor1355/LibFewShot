# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core import Trainer
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-arch', required=False, type=str, help='architecture to use')
    args = parser.parse_args()
    
    if args.arch == "relationbert":
        config = Config("./config/relationbert.yaml").get_config_dict()
    else:
        config = Config("./config/relationbert.yaml").get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
