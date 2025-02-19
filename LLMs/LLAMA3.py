from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os


class LLAMA3(nn.Module):
    def __init__(self, args):
        super(LLAMA3, self).__init__()
        self.args = args
        if not args.train:
            # device_map = {
            #     "language_model": 1,
            #     "language_projection": 2
            # }
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                           token='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')
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                              token='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',
                                                              device_map="auto",
                                                              quantization_config=quantization_config)
                                                              # device_map=device_map)
            # self.model = self.model.to(device='cuda:1')
            if args.fine_tuned:
                self.model = PeftModel.from_pretrained(self.model, os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint-4350/'))
        # else:
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     # bnb_4bit_quant_type="nf4",
            #     # bnb_4bit_use_double_quant=True,
            #     bnb_8bit_compute_dtype=torch.bfloat16
            # )
            # self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
            #                                                   device_map='auto',
            #                                                   quantization_config=bnb_config)
            # self.model.gradient_checkpointing_enable()
            # if torch.cuda.device_count() > 1:
            #     self.model.is_parallelizable = True
            #     self.model.model_parallel = True

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            # inputs = self.tokenizer(inputs, return_tensors="pt").to(device='cuda:1')
            generated_ids = self.model.generate(**inputs, max_new_tokens=250)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            # output = output.replace('</s>', '')
            # output = output.replace("['", '')
            # output = output.replace("']", '')
            # output = output.replace('["', '')
            # output = output.replace('"]', '')
            output = output.split(':')[-1]
            print(output)
            return output
        # return self.model(**inputs)
