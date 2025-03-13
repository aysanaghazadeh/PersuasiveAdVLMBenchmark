from utils.data.data_util import load_data
from LLMs.LLM import LLM
import json
import os
from jinja2 import Environment, FileSystemLoader


def generate_hard_qa_question(args, clean_set, prompt_file):
    f = open(os.path.join(args.data_path, args.test_set_QA))
    question_answers = json.load(f)
    new_QA = {}
    for i, question in enumerate(question_answers):
        if not (question in clean_set):
            continue
        new_QA[question] = [question_answers[question][0][0:3]]
        new_choices = set(question_answers[question][0][0:3])
        correct_options = question_answers[question][0][0:3]
        for c in correct_options:
            env = Environment(loader=FileSystemLoader(args.prompt_path))
            template = env.get_template(prompt_file)
            data = {'correct_statement': c}
            prompt = template.render(**data)
            negation = LLM(prompt)
            print(c, '-->', negation)
            if ':' in negation:
                new_choices.add(negation.split(':')[1].strip())
            else:
                new_choices.add(negation.strip())
        # print(new_choices)
        new_QA[question].append(list(new_choices))
    return new_QA


def generate_all_hard_qa(args):
    clean_set = load_data(args)

