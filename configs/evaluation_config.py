import torch
import yaml
import argparse
from transformers import set_seed


def read_yaml_config(file_path):
    """Reads a YAML configuration file and returns a dictionary of the settings."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def convert_to_args(config):
    """Converts a nested dictionary to a list of command-line arguments."""

    args = {}
    for section, settings in config.items():
        for key, value in settings.items():
            args[key] = value
    return args


def set_conf(config_file):
    yaml_file_path = config_file
    config = read_yaml_config(yaml_file_path)
    args = convert_to_args(config)
    return args


def parse_args():
    """ Parsing the Arguments for the Advertisement Generation Project"""
    parser = argparse.ArgumentParser(description="Configuration arguments for Persuasive Ad VLM Benchmark:")
    #config
    parser.add_argument('--config_type',
                        type=str,
                        required=True,
                        help='Choose among ARGS for commandline arguments, DEFAULT for default values, or YAML for '
                             'config file')
    parser.add_argument('--config_path',
                        type=str,
                        default=None,
                        help='The path to the config file if config_type is YAML')
    # task
    parser.add_argument('--task',
                        type=str,
                        default='PittAd',
                        choices=['whoops', 'PittAd'],
                        help='Choose between PittAd, whoops')
    parser.add_argument('--top_k',
                        type=int,
                        default=3,
                        help='if multi-selection select the k, if single select 1')
    # data
    parser.add_argument('--data_path',
                        type=str,
                        default='../Data/PittAd',
                        help='Path to the root of the data'
                        )
    parser.add_argument('--train_set_QA',
                        type=str,
                        default='train/QA_Combined_Action_Reason_train.json',
                        help='If the model is fine-tuned, relative path to the train-set QA from root path')
    parser.add_argument('--train_set_images',
                        type=str,
                        default='train_images',
                        help='If the model is fine-tuned, relative path to the train-set Images from root path')
    parser.add_argument('--test_set_QA',
                        type=str,
                        default='train/QA_Combined_Action_Reason_train.json',
                        help='Relative path to the QA file for action-reasons from root path'
                        )
    parser.add_argument('--test_set_images',
                        type=str,
                        default='train_images_total',
                        help='Relative path to the original images for the test set from root')
    # context
    parser.add_argument('--description_type',
                        type=str,
                        default=None,
                        choices=['combine', 'IN', 'UH', 'V', 'T'],
                        help='Choose among IN, UH, combine')
    parser.add_argument('--description_file',
                        type=str,
                        default=None)
    parser.add_argument('--object_file',
                        type=str,
                        default=None)
    parser.add_argument('--with_atypicality',
                        action="store_true",
                        default=None)
    parser.add_argument('--atypicality_file',
                        type=str,
                        default=None)
    # models
    parser.add_argument('--method',
                        type=str,
                        choices=['LLM', 'VLM'],
                        default=None)
    parser.add_argument('--LLM',
                        type=str,
                        default=None)
    parser.add_argument('--VLM',
                        type=str,
                        default=None)
    # prompts
    parser.add_argument('--prompt_path',
                        type=str,
                        default='util/prompt_engineering/prompts',
                        help='Path to the folder of prompts. Set the name of prompt files as: {text_input_type}.jinja')
    parser.add_argument('--VLM_prompt',
                        type=str,
                        default=None)
    parser.add_argument('--LLM_prompt',
                        type=str,
                        default=None)
    parser.add_argument('--format_prompt',
                        type=str,
                        default=None)
    # results
    parser.add_argument('--result_path',
                        type=str,
                        default='../experiments/results',
                        help='The path to the folder for saving the results')
    parser.add_argument('--result_file',
                        type=str,
                        default=None,
                        help='the file path relative to the result_path')

    return parser.parse_args()


def get_args():
    set_seed(0)
    args = parse_args()
    if args.config_type == 'YAML':
        args = set_conf(args.config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    print("Arguments are:\n", args, '\n', '-'*40)
    return args




