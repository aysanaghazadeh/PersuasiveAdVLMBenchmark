from torch import nn
from VLMs.LLAVA import LLAVA
from VLMs.InternVL2 import InternVL
from VLMs.LLAVA16 import LLAVA16


class VLM(nn.Module):
    def __init__(self, args):
        super(VLM, self).__init__()
        model_map = {
            'LLAVA': LLAVA,
            'LLAVA16': LLAVA16,
            'InternVL': InternVL
        }
        self.model = model_map[args.VLM](args)

    def forward(self, prompt):
        output = self.model(prompt)
        return output

