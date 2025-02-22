from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class LLAMA3(nn.Module):
    def __init__(self, args):
        super(LLAMA3, self).__init__()
        self.args = args
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                          device_map="auto",
                                                          quantization_config=quantization_config)

    def forward(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=250)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        output = output.split(':')[-1]
        print(output)
        return output
