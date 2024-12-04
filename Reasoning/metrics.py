import json
from jinja2 import Environment, FileSystemLoader
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import functional as TF
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import tempfile
from transformers import pipeline, BitsAndBytesConfig
import re
import base64
import requests
from VLMs.InternVL2 import InternVL
from VLMs.multi_image_InternVL import MultiInternVL
import itertools
from LLMs.LLM import LLM
from FlagEmbedding import BGEM3FlagModel

api_key = "sk-proj-zfkbSHxUNuF7Ev8TEWWRT3BlbkFJieFKktR5T8tIUVNAJRBz"


# Function to convert an image file to a tensor
def image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
    return tensor

class Whoops:
    def __init__(self, args):
        self.args = args
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            torch_dtype="float16"
        )
        model_id_map = {
            'LLAVA': "llava-hf/llava-1.5-13b-hf",
            'LLAVA_phi': "xtuner/llava-phi-3-mini-hf",
            'VILA': "Efficient-Large-Model/VILA-7b",
            'InternVL': "OpenGVLab/InternVL-Chat-V1-5"
        }
        model_id = model_id_map[args.VLM]
        task_map = {
            'LLAVA': "image-to-text",
            'LLAVA_LLAMA': "image-to-text",
            'VILA': "text-generation",
            'InternVL': "visual-question-answering"
        }
        task = task_map[args.VLM]
        self.pipe = pipeline(task,
                             model=model_id,
                             model_kwargs={"quantization_config": quantization_config},
                             trust_remote_code=True,
                             device_map='auto',
                             token='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', )
        self.QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))

    @staticmethod
    def parse_options(options):
        return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

    @staticmethod
    def get_answer_format():
        answer_format = """
        Answer: ${index of the best option}\n
        """
        return answer_format

    def get_prompt(self, options, question=None, description=None):
        options = self.parse_options(options)
        data = {'options': options, 'question': question, 'description': description}
        env = Environment(loader=FileSystemLoader(self.args.prompt_path))
        template = env.get_template(self.args.VLM_prompt)
        prompt = template.render(**data)
        return prompt

    def get_prediction(self, image, description, QA):
        answers = []
        options = QA[-1]
        if len(QA) == 3:
            question = QA[0]
        else:
            question = None
        prompt = self.get_prompt(options, question, description)
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 45})
        output = output[0]["generated_text"].split(':')[-1]
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
        return answers
