import sys
import os
import csv
import argparse

current_dir = os.path.abspath(os.getcwd())
sys.path.append(current_dir)
from novobench.datasets import CustomDataset, NineSpeciesDataset
from novobench.models.imputation_denovo import ImpnovoRunner
from novobench.utils.config import Config



def train(config_file, data_dir, model_file):
    config = Config(config_file, "impnovo")
    file_mapping = {
    "train" : "train.parquet",
    "valid" : "valid.parquet",
    }
    dataset = CustomDataset(data_dir, file_mapping)
    data = dataset.load_data(transform = ImpnovoRunner.preprocessing_pipeline(config))
    model = ImpnovoRunner(config, model_file)
    model.train(data.get_train(), data.get_valid())




def denovo(config_file,data_dir, model_file, saved_path):
    config = Config(config_file, "impnovo")
    file_mapping = {"valid" : "test.parquet",}
    dataset = CustomDataset(data_dir, file_mapping)
    data = dataset.load_data(transform = ImpnovoRunner.preprocessing_pipeline(config))
    model = ImpnovoRunner(config, model_file, saved_path)
    model.denovo(data.get_valid())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--data_path", type=str,required=True)
    parser.add_argument("--ckpt_path", type=str,default=None)
    parser.add_argument("--denovo_output_path", type=str,default='')
    parser.add_argument("--config_path", type=str,required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.config_path, args.data_path, args.ckpt_path)
    elif args.mode == "denovo":
        denovo(args.config_path, args.data_path, args.ckpt_path, args.denovo_output_path) 
    else:
        raise ValueError("Invalid mode!")

        