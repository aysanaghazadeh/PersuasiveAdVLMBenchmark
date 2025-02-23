import json
import pandas as pd
from PIL import Image
import os
from VLMs.VLM import VLM
from LLMs.LLM import LLM
from jinja2 import Environment, FileSystemLoader
import csv


class Retrieval:
    """ARR and ASR tasks."""
    def __init__(self, args):
        self.pipe = VLM(args) if args.method == 'VLM' else LLM(args)
        self.descriptions = None
        self.atypicalities = None
        self.args = args
        self.QAs = json.load(open(os.path.join(self.args.data_path, self.args.test_set_QA)))
        self.set_descriptions()
        self.set_atypicalities()

    @staticmethod
    def parse_options(options):
        return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

    def get_answer_format(self):
        env = Environment(loader=FileSystemLoader(self.args.prompt_path))
        template = env.get_template(self.args.format_prompt)
        answer_format = template.render()
        return answer_format

    def get_prompt(self, options, answer_format, description, question=None, atypicality=None):
        data = {'options': options,
                'answer_format': answer_format,
                'description': description,
                'question': question,
                'atypicality': atypicality}
        env = Environment(loader=FileSystemLoader(self.args.prompt_path))
        prompt_file = self.args.VLM_prompt if self.args.method == 'VLM' else self.args.LLM_prompt
        template = env.get_template(prompt_file)
        prompt = template.render(**data)
        return prompt

    def set_descriptions(self):
        if self.args.description_file is not None:
            self.descriptions = pd.read_csv(self.args.description_file)

    def set_atypicalities(self):
        if self.args.with_atypicality:
            self.atypicalities = pd.read_csv(self.args.atypicality_file)

    def get_description(self, image_url):
        description = self.descriptions.loc[self.descriptions['ID'] == image_url].iloc[0]['description']
        return description

    def get_atypicality(self, image_url):
        atypicality = self.atypicalities.loc[self.atypicalities['ID'] == image_url].iloc[0]['description']
        return atypicality

    def get_image(self, image_url):
        image_path = os.path.join(self.args.data_path, self.args.test_set_images, image_url)
        image = Image.open(image_path)
        return image

    def get_predictions_PittAd(self, image_url):
        options = self.QAs[image_url][1]
        options_formatted = self.parse_options(options)
        description = self.get_description(image_url)
        answer_format = self.get_answer_format()
        atypicality = self.get_atypicality(image_url) if self.args.with_atypicality else None
        prompt = self.get_prompt(options_formatted, answer_format, description, atypicality)
        image = self.get_image(image_url)
        if self.args.method == 'VLM':
            output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 45})
        else:
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
        atypicality = self.get_atypicality(image_url) if self.args.with_atypicality else None
        prompt = self.get_prompt(options_formatted, answer_format, description, question, atypicality)
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
        correct_options = self.QAs[image_url][0] if self.args.task == 'PittAd' else self.QAs[image_url][1]
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

    def get_all_results(self):
        results = {}
        for i in range(0, self.args.top_k):
            results[f'acc@{i + 1}'] = 0
        fieldnames = ['id']
        for i in range(0, self.args.top_k):
            fieldnames.append(f'acc@{i + 1}')
        csv_file_path = os.path.join(self.args.result_path,
                                     f'{self.args.task}'
                                     f'_{self.args.test_set_QA.split("/")[-1].replace(".json", "")}'
                                     f'_{self.args.method}'
                                     f'_{self.args.description_file.split("/")[-1].replace(".csv", "") if self.args.description_file else ""}'
                                     f'_{self.args.VLM_prompt.replace(".jinja", "") if self.args.method == "VLM" else self.args.LLM_prompt.replace(".jinja", "")}'
                                     f'.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        for image_url in self.QAs:
            image_url = f'{image_url}.png' if '.' not in image_url else image_url
            result = self.reason_image(image_url)
            row = {}
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row['id'] = image_url
                for key in result:
                    row[key] = result[key]
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(self.QAs.keys()))}')

