import json
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
from PIL import Image
import os
from VLMs.VLM import VLM


class ActionReasonVLM:
    def __init__(self, args):
        self.pipe = VLM(args)
        self.descriptions = None
        self.args = args
        self.QAs = json.load(open(os.path.join(self.args.data_path, self.args.test_set_QA)))
        self.set_descriptions()

    @staticmethod
    def parse_options(options):
        return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

    @staticmethod
    def get_answer_format():
        # answer_format = """
        # Answer: ${indices of three best options}\n
        # """
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.format_prompt)
        answer_format = template.render()
        return answer_format

    @staticmethod
    def get_prompt(options, answer_format, description):
        # prompt = (f"USER:<image>\n"
        #  f"Question: What are the indices of the 3 best interpretations in ranked form among the options for this image? Separate them by comma.\n"
        #  f"Options: {options}\n"
        #  f"your answer must only follow the format of {answer_format} not any other format.\n"
        #  f"Do not return include any explanation, only the indices of the three best options separated by comma."
        #  f"Assistant: ")
        data = {'options': options,
                'answer_format': answer_format,
                'description': description}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.VLM_prompt)
        prompt = template.render(**data)
        return prompt

    def set_descriptions(self):
        if self.args.description_file is not None:
            self.descriptions = pd.read_csv(self.args.description_file)

    def get_description(self, image_url):
        description = self.descriptions.loc[self.descriptions['ID'] == image_url].iloc[0]['description']
        return description

    def get_image(self, image_url):
        image_path = os.path.join(self.args.data_path, self.args.test_set_images, image_url)
        image = Image.open(image_path)
        return image

    def get_predictions(self, image_url):
        options = self.QAs[image_url][1]
        options_formatted = self.parse_options(options)
        description = self.get_description(image_url)
        answer_format = self.get_answer_format()
        prompt = self.get_prompt(options_formatted, answer_format, description)
        image = self.get_image(image_url)
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 45})
        if self.args.VLM == 'LLAVA':
            output = output[0]["generated_text"]
        print(output)
        options = self.QAs[image_url][1]
        predictions = output.split(',')
        answers = []
        for output in predictions:
            answer = ''.join(i for i in output if i.isdigit())
            if answer != '':
                answers.append(int(answer))
        predictions = set()
        for ind in answers:
            if len(options) > ind:
                predictions.add(options[ind])
                if len(predictions) == 3:
                    break
        answers = list(predictions)
        print(f'predictions for image {image_url} is {answers}')
        return answers

    def evaluate_answers(self, image_url, answers):
        correct_options = self.QAs[image_url][0]
        results = {}
        for i in range(3):
            count = 0
            for answer in answers[:i+1]:
                if answer in correct_options:
                    count += 1
            results[f'acc@{i + 1}'] = min(1, count / 1)
        for i in range(3):
            count = 0
            for answer in answers[:3]:
                if answer in correct_options:
                    count += 1
            results[f'p@{i + 1}'] = min(1, count / (i + 1))
        print(f'results for image {image_url} is {results}')
        return results

    def reason_image(self, image_url):
        answers = self.get_predictions(image_url)
        evaluation_results = self.evaluate_answers(image_url, answers)
        return evaluation_results


