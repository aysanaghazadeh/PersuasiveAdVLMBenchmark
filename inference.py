import os.path
import pandas as pd
from collections import Counter
import requests
from configs.evaluation_config import get_args
from Reasoning.retrieval import ActionReasonVLM
import csv
from util.data.trian_test_split import get_test_data
from LLMs.LLM import LLM
from io import BytesIO
from Reasoning.metrics import *
from PIL import Image


class Reasoning:
    def __init__(self, args):
        if args.evaluation_type == 'action_reason_VLM':
            self.ar_VLM = ActionReasonVLM(args)
        if args.evaluation_type == 'whoops_llava':
            self.whoops = Whoops(args)

    def reason_action_reason_VLM(self, args):
        results = {'acc@1': 0, 'acc@2': 0, 'acc@3': 0,
                   'p@1': 0, 'p@2': 0, 'p@3': 0}
        fieldnames = ['acc@1', 'acc@2', 'acc@3', 'p@1', 'p@2', 'p@3', 'id']
        # csv_file_path = os.path.join(args.result_path, ''.join(['action_reason_llava_', args.description_file]))
        csv_file_path = os.path.join(args.result_path, f'action_reason_{args.VLM}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        for image_url in QAs:
            result = self.ar_VLM.reason_image(image_url)
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = result
                row['id'] = image_url
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')

    def reason_whoops_VLM(self, args):
        results = {}
        for i in range(0, args.top_k):
            results[f'acc@{i + 1}'] = 0
        fieldnames = ['id']
        for i in range(0, args.top_k):
            fieldnames.append(f'acc@{i + 1}')
        csv_file_path = os.path.join(args.result_path, f'{args.test_set_QA.split("/")[-1].replace(".json", "")}'
                                                       f'_{args.description_type}'
                                                       f'_{args.VLM}_description_{args.LLM}_'
                                                       f'{args.VLM_prompt.replace(".jinja", "")}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QA_file = os.path.join(args.data_path, args.test_set_QA)
        QAs = json.load(open(QA_file))
        descriptions = pd.read_csv(os.path.join(args.data_path,
                                                'train',
                                                f'{args.description_type}_{args.VLM}_description_{args.task}.csv'))
        for image_index in QAs:
            image_url = f'{image_index}.png'
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
            image = Image.open(os.path.join(args.data_path, 'whoops_images', image_url))
            answers = self.whoops.get_prediction(image, description, QAs[image_index])
            correct_options = QAs[image_index][1] if len(QAs[image_index][0]) == 3 else QAs[image_index][0]
            print(answers)
            if len(answers) == 0:
                result = {}
                for i in range(0, args.top_k):
                    result[f'acc@{i + 1}'] = 0
            else:
                result = {}
                for i in range(0, args.top_k):
                    result[f'acc@{i + 1}'] = 0
                for i, answer in enumerate(answers[0: args.top_k]):
                    if answer in correct_options:
                        for j in range(i, args.top_k):
                            result[f'acc@{j + 1}'] = 1
            print(result)
            row = {}
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row['id'] = i
                for metric in result:
                    row[metric] = result[metric]
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')

    @staticmethod
    def reason_whoops_LLM(args):
        def parse_options(options):
            return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

        def get_prediction(prompt, options, pipe):
            answers = []
            output = pipe(prompt)
            print(output)
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

        results = {}
        for i in range(0, args.top_k):
            results[f'acc@{i + 1}'] = 0
        fieldnames = ['id']
        for i in range(0, args.top_k):
            fieldnames.append(f'acc@{i + 1}')
        csv_file_path = os.path.join(args.result_path, f'{args.test_set_QA.split("/")[-1].replace(".json", "")}'
                                                       f'_{args.description_type}'
                                                       f'_{args.VLM}_description_{args.LLM}_'
                                                       f'{args.VLM_prompt.replace(".jinja", "")}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QA_file = os.path.join(args.data_path, args.test_set_QA)
        QAs = json.load(open(QA_file))
        pipe = LLM(args)
        descriptions = pd.read_csv(
            os.path.join(args.data_path, 'train', f'{args.description_type}_{args.VLM}_description_{args.task}.csv'))
        for image in QAs:
            image_url = f'{image}.png'
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
            options = QAs[image][1]
            env = Environment(loader=FileSystemLoader(args.prompt_path))
            template = env.get_template(args.VLM_prompt)
            data = {'description': description, 'options': parse_options(options)}
            prompt = template.render(**data)
            answers = get_prediction(prompt, options, pipe)
            correct_options = QAs[image][1] if len(QAs[image][0]) == 3 else QAs[image][0]
            print(answers)
            if len(answers) == 0:
                result = {}
                for i in range(0, args.top_k):
                    result[f'acc@{i + 1}'] = 0
            else:
                result = {}
                for i in range(0, args.top_k):
                    result[f'acc@{i + 1}'] = 0
                for i, answer in enumerate(answers[0: args.top_k]):
                    if answer in correct_options:
                        for j in range(i, args.top_k):
                            result[f'acc@{j + 1}'] = 1
            print(result)
            row = {}
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row['id'] = i
                for metric in result:
                    row[metric] = result[metric]
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')

    def reason_action_reason_LLM(self, args):
        def parse_options(options):
            return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

        def get_prediction(prompt, options, pipe):
            answers = []
            output = pipe(prompt)
            print(output)
            outputs = output.split(',')
            predictions = [''.join(i for i in output if i.isdigit()) for output in outputs]
            print(predictions)
            for answer in predictions:
                if answer != '':
                    answers.append(int(answer))
            predictions = set()
            for ind in answers:
                if len(options) > ind:
                    predictions.add(options[ind])
                    if len(predictions) == 3:
                        break
            answers = list(predictions)
            return answers

        results = {'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'p@1': 0, 'p@2': 0, 'p@3': 0}
        fieldnames = ['acc@1', 'acc@2', 'acc@3', 'p@1', 'p@2', 'p@3', 'id', 'prediction']
        csv_file_path = os.path.join(args.result_path,
                                     f'PittAd_{args.description_type}_{args.VLM}_description_{args.LLM}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QA_file = os.path.join(args.data_path, args.test_set_QA)
        QAs = json.load(open(QA_file))
        pipe = LLM(args)
        descriptions = pd.read_csv(args.description_file)
        for image_url in QAs:
            if image_url not in descriptions.ID.values:
                continue
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
            print('description:', description)
            options = QAs[image_url][1]
            correct_options = QAs[image_url][0]
            env = Environment(loader=FileSystemLoader(args.prompt_path))
            template = env.get_template(args.VLM_prompt)
            data = {'description': description, 'options': parse_options(options)}
            prompt = template.render(**data)
            answers = get_prediction(prompt, options, pipe)
            result = {'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'p@1': 0, 'p@2': 0, 'p@3': 0}
            print(answers)
            correct_count = 0
            if len(answers) != 0:
                for i, answer in enumerate(answers[0:3]):
                    if answer in correct_options:
                        correct_count += 1
                        for j in range(i, 3):
                            result[f'acc@{j + 1}'] = 1
                result['p@1'] = min(correct_count, 1)
                result['p@2'] = min(correct_count / 2, 1)
                result['p@3'] = min(correct_count / 3, 1)

            print(result)
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = result
                row['id'] = image_url
                row['prediction'] = answers
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')

    def reason(self, args):
        evaluation_name = 'reason_' + args.evaluation_type
        print(f'evaluation method: {evaluation_name}')
        evaluation_method = getattr(self, evaluation_name)
        evaluation_method(args)


if __name__ == '__main__':
    args = get_args()
    reasoning = Reasoning(args)
    reasoning.reason(args)
