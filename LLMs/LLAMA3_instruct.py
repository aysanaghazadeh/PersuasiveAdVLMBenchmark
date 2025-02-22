from torch import nn
from transformers import pipeline
import torch


class LLAMA3Instruct(nn.Module):
    def __init__(self, args):
        super(LLAMA3Instruct, self).__init__()
        self.args = args
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def forward(self, prompt):

        messages = [
            {"role": "system", "content": "Be a helpful assistant"},
            {"role": "user", "content": prompt},
        ]
        output = self.pipeline(messages, max_new_tokens=250)
        output = output[0]["generated_text"][-1]['content'].split('ASSISTANT:')[-1]
        print('llama3 output:', output)
        return output
