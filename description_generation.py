import json
from jinja2 import Environment, FileSystemLoader
from transformers import pipeline
from util.data.trian_test_split import get_test_data, get_train_data
from PIL import Image
import os
import csv
import pandas as pd
from configs.inference_config import get_args
from util.prompt_engineering.prompt_generation import PromptGenerator
from LLMs.LLM import LLM
from VLMs.VLM import VLM


def get_model(args):
    if args.description_type == 'combine':
        pipe = LLM(args)
        return pipe
    else:
        pipe = VLM(args)
        return pipe


def get_llm(args):
    model = PromptGenerator(args)
    model.set_LLM(args)
    return model


def get_single_description(args, image_url, pipe):
    image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
    env = Environment(loader=FileSystemLoader(args.prompt_path))
    template = env.get_template(args.VLM_prompt)
    prompt = template.render()
    description = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})
    return description


def get_combine_description(args, image_url, pipe):
    IN_descriptions = pd.read_csv(os.path.join(args.result_path,
                                               f'IN'
                                               f'_{args.VLM}'
                                               f'_description_{args.task}.csv'))
    IN_description = IN_descriptions.loc[IN_descriptions['ID'] == image_url]['description'].values[0]
    UH_descriptions = pd.read_csv(os.path.join(args.result_path,
                                               f'UH'
                                               f'_{args.VLM}'
                                               f'_description_{args.task}.csv'))
    UH_description = UH_descriptions.loc[UH_descriptions['ID'] == image_url]['description'].values[0]
    v_descriptions = pd.read_csv(os.path.join(args.result_path,
                                              f'V'
                                              f'_{args.VLM}'
                                              f'_description_{args.task}.csv'))
    v_description = v_descriptions.loc[v_descriptions['ID'] == image_url]['description'].values[0]
    T_descriptions = pd.read_csv(os.path.join(args.result_path,
                                              f'T'
                                              f'_{args.VLM}'
                                              f'_description_{args.task}.csv'))
    T_description = T_descriptions.loc[T_descriptions['ID'] == image_url]['description'].values[0]
    data = {'IN': IN_description, 'UH': UH_description, 'v': v_description, 'T': T_description, 'token_length': None}
    env = Environment(loader=FileSystemLoader(args.prompt_path))
    template = env.get_template(args.LLM_prompt)
    prompt = template.render(**data)
    description = pipe(prompt=prompt)
    return description


def get_descriptions(args):
    if args.task == 'whoops':
        images = [f'{i}.png' for i in range(500)]
    else:
        images = list(json.load(open(os.path.join(args.data_path, args.test_set_QA))).keys())

    print(f'number of images in the set: {len(images)}')
    print('*' * 100)

    description_file = os.path.join(args.result_path,
                                    f'{args.description_type}'
                                    f'_{args.VLM}'
                                    f'_description_{args.task}.csv')

    if os.path.exists(description_file):
        print(description_file)
        return pd.read_csv(description_file)
    with open(description_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    pipe = get_model(args)
    processed_images = set()
    for image_url in images:
        if image_url in processed_images:
            continue
        processed_images.add(image_url)
        if args.description_type == 'combine':
            description = get_combine_description(args, image_url, pipe)
        else:
            description = get_single_description(args, image_url, pipe)
        print(f'output of image {image_url} is {description}')
        print('-' * 80)
        pair = [image_url, description]
        with open(description_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

    return pd.read_csv(description_file)


if __name__ == '__main__':
    args = get_args()
    descriptions = get_descriptions(args)
