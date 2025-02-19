from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch


class LLAVA(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipe = pipeline("image-to-text", model='llava-hf/llava-1.5-13b-hf', device_map='auto')

    def forward(self, image, prompt, generate_kwargs={"max_new_tokens": 250}):
        outputs = self.pipe(image, prompt=prompt, generate_kwargs=generate_kwargs)
        output = outputs[0]['generated_text'].split('ASSISTANT: ')[-1]
        return output