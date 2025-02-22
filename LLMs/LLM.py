from torch import nn
from LLMs.LLAMA3 import LLAMA3
from LLMs.vicuna import Vicuna
from LLMs.LLAMA3_instruct import LLAMA3Instruct
from LLMs.InternLM import InternLM


class LLM(nn.Module):
    def __init__(self, args):
        super(LLM, self).__init__()
        model_map = {
            'LLAMA3': LLAMA3,
            'LLAMA3_instruct': LLAMA3Instruct,
            'vicuna': Vicuna,
            'InternLM': InternLM
        }
        self.model = model_map[args.LLM](args)

    def forward(self, prompt):
        output = self.model(prompt)
        return output

