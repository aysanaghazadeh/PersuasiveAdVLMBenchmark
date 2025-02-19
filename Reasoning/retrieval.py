import json
import pandas as pd
from PIL import Image
import os
from VLMs.VLM import VLM
from jinja2 import Environment, FileSystemLoader


class RetrievalVLM:
    def __init__(self, args):
        self.pipe = VLM(args)
        self.descriptions = None
        self.args = args
        self.QAs = json.load(open(os.path.join(self.args.data_path, self.args.test_set_QA)))
        self.set_descriptions()

    @staticmethod
    def parse_options(options):
        return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

    def get_answer_format(self):
        # answer_format = """
        # Answer: ${indices of three best options}\n
        # """
        env = Environment(loader=FileSystemLoader(self.args.prompt_path))
        template = env.get_template(self.args.format_prompt)
        answer_format = template.render()
        return answer_format

    def get_prompt(self, options, answer_format, description, question=None):
        # prompt = (f"USER:<image>\n"
        #  f"Question: What are the indices of the 3 best interpretations in ranked form among the options for this image? Separate them by comma.\n"
        #  f"Options: {options}\n"
        #  f"your answer must only follow the format of {answer_format} not any other format.\n"
        #  f"Do not return include any explanation, only the indices of the three best options separated by comma."
        #  f"Assistant: ")
        data = {'options': options,
                'answer_format': answer_format,
                'description': description,
                'question': question}
        env = Environment(loader=FileSystemLoader(self.args.prompt_path))
        template = env.get_template(self.args.VLM_prompt)
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

    def get_predictions_PittAd(self, image_url):
        options = self.QAs[image_url][1]
        options_formatted = self.parse_options(options)
        description = self.get_description(image_url)
        answer_format = self.get_answer_format()
        prompt = self.get_prompt(options_formatted, answer_format, description)
        image = self.get_image(image_url)
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 45})
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

    def get_predictions_Whoops(self, image_url):
        answers = []
        options = self.QAs[image_url][-1]
        options_formatted = self.parse_options(options)
        description = self.get_description(image_url)
        answer_format = self.get_answer_format()
        if len(self.QAs[image_url]) == 3:
            question = self.QAs[image_url][0]
        else:
            question = None
        prompt = self.get_prompt(options_formatted, answer_format, description, question)
        image = self.get_image(image_url)
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 45})
        predictions = [''.join(i for i in prediction if i.isdigit()) for prediction in output.split(',')]
        for prediction in predictions:
            if prediction != '':
                answers.append(int(prediction))
        predictions = set()
        for ind in answers:
            if len(options) > ind:
                predictions.add(options[ind])
        answers = list(predictions)
        return answers

    def evaluate_answers(self, image_url, answers):
        correct_options = self.QAs[image_url][0]
        results = {'prediction': ','.join(answers)}
        if self.args.top_k == 1:
            results['accuracy'] = 1 if answers[0] in correct_options else 0
        for i in range(3):
            count = 0
            for answer in answers[:i + 1]:
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
        prediction_function = 'get_predictions_' + self.args.task
        print(f'evaluation method: {prediction_function}')
        get_prediction = getattr(self, prediction_function)
        get_prediction(self.args)
        answers = self.get_predictions_PittAd(image_url)
        evaluation_results = self.evaluate_answers_PittAd(image_url, answers)
        return evaluation_results
