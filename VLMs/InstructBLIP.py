from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from torch import nn


class InstructBLIP(nn.Module):
    def __init__(self, args):
        super(InstructBLIP, self).__init__()
        self.args = args
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.model.to(args.device)

    def forward(self, prompt, image):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.args.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return output