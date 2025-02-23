import json
from jinja2 import Environment, FileSystemLoader
from PIL import Image
import os
import csv
import pandas as pd
from configs.evaluation_config import get_args
from LLMs.LLM import LLM
from VLMs.VLM import VLM
from utils.mapping.atypicality_maps import *
import re


def get_model(args):
    if args.description_type in ['combine', 'atypicality']:
        pipe = LLM(args)
        return pipe
    else:
        pipe = VLM(args)
        return pipe


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


def get_atypicality(v_description, description, pipe, image_url):
    def get_objects():
        objects = re.split(r'\d.', v_description)
        objects.remove('')
        return objects
    def generate_options(objects):
        options = set()
        for o1 in objects:
            for relation in atypicality_def:
                for o2 in objects:
                    if o1 == o2:
                        continue
                    relation_statement = atypicality_def[relation].format(primary_concept=o1, secondary_concept=o2)
                    options.add(relation_statement)
        options = list(options)
        return options

    def parse_options(options):
        return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

    def get_answer_format():
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.format_prompt)
        answer_format = template.render()
        return answer_format

    def get_prompt(options_formatted, answer_format, description):
        data = {'options': options_formatted,
                'answer_format': answer_format,
                'description': description}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        prompt_file = args.VLM_prompt if args.method == 'VLM' else args.LLM_prompt
        template = env.get_template(prompt_file)
        prompt = template.render(**data)
        return prompt

    objects = get_objects()
    options = generate_options(objects)
    answer_format = get_answer_format()
    options_formatted = parse_options(options)
    prompt = get_prompt(options_formatted, answer_format, description)
    image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
    if args.method == 'VLM':
        output = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 45})
    else:
        output = pipe(prompt=prompt, generate_kwargs={"max_new_tokens": 45})
    predictions = output.split(',')
    answers = []
    for output in predictions[args.top_k]:
        answer = ''.join(i for i in output if i.isdigit())
        if answer != '':
            answers.append(int(answer))
    predictions = set()
    for ind in answers:
        if len(options) > ind:
            predictions.add(options[ind])
            if len(predictions) == 3:
                break
    answers = ', '.joing(list(predictions))
    print(f'predictions for image {image_url} is {answers}')
    return answers


def get_descriptions(args):
    if args.task == 'whoops':
        images = [f'{i}.png' for i in range(500)]
    else:
        images = list(json.load(open(os.path.join(args.data_path, args.test_set_QA))).keys())

    print(f'number of images in the set: {len(images)}')
    print('*' * 100)
    if args.description_type == 'atypicality':
        description_file = os.path.join(args.result_path,
                                        f'{args.description_type}'
                                        f'_{args.LLM}'
                                        f'_{args.description_file.split("/")[-1].split("_")[0]}'
                                        f'_description.csv')
        v_descriptions = pd.read_csv(args.object_file)
        descriptions = pd.read_csv(args.description_file)
    else:
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
        elif args.description_type == 'atypicality':
            v_description = v_descriptions.loc[v_descriptions['ID'] == image_url].iloc[0]['description']
            context_description = descriptions.loc[descriptions['ID'] == image_url].iloc[0]['description']
            description = get_atypicality(v_description, context_description, pipe, image_url)
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
