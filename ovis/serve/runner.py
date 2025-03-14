from dataclasses import field, dataclass
from typing import Optional, Union, List

import torch
from PIL import Image
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch import nn

from ovis.model.modeling_ovis import Ovis
from ovis.util.constants import IMAGE_TOKEN

@dataclass
class RunnerArguments:
    model_path: str
    max_new_tokens: int = field(default=512)
    do_sample: bool = field(default=False)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    temperature: Optional[float] = field(default=None)
    max_partition: int = field(default=9)

class OvisRunner:
    def __init__(self, args: RunnerArguments):
        self.model_path = args.model_path
        self.dtype = torch.bfloat16
        self.device = torch.cuda.current_device()

        # Load model with accelerate's device_map for proper parallelism
        self.model = Ovis.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,  # Reduce memory usage during loading
            # Use accelerate's balanced device_map
            device_map=self._get_balanced_device_map(self.model)
        )
        self.model.eval()  # Ensure evaluation mode

        # Ensure all parameters are on the correct devices
        dispatch_model(self.model, device_map=self.model.device_map)

        self.eos_token_id = self.model.generation_config.eos_token_id
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()

        self.image_placeholder = IMAGE_TOKEN
        self.max_partition = args.max_partition
        self.gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )

    def _get_balanced_device_map(self, model):
        # Use accelerate's balanced memory allocation
        max_memory = get_balanced_memory(
            model,
            max_memory={f"cuda:{i}": "5GB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["VisualTransformerEmbedding"]  # Adjust based on your model
        )
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["VisualTransformerEmbedding"]
        )
        return device_map

    def preprocess(self, inputs: List[Union[Image.Image, str]]):
        # Ensure image is first if mixed with text
        if len(inputs) == 2 and isinstance(inputs[0], str) and isinstance(inputs[1], Image.Image):
            inputs = reversed(inputs)

        query = ''
        images = []
        for data in inputs:
            if isinstance(data, Image.Image):
                query += self.image_placeholder + '\n'
                images.append(data)
            elif isinstance(data, str):
                query += data.replace(self.image_placeholder, '')
            else:
                raise RuntimeError(f"Invalid input type: {type(data)}")

        # Preprocess with model's method (handles device placement via device_map)
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=self.max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)

        # Ensure tensors are on the correct devices
        input_ids = input_ids.unsqueeze(0).to(self.model.device_map["text_encoder"])  # Example: adjust based on your model's device_map
        attention_mask = attention_mask.unsqueeze(0).to(self.model.device_map["text_encoder"])

        if pixel_values is not None:
            # Move pixel_values to the correct device (e.g., the first GPU)
            pixel_values = [pv.to(device=self.model.device_map["visual_encoder"], dtype=self.dtype) for pv in pixel_values]
        else:
            pixel_values = [None]

        return prompt, input_ids, attention_mask, pixel_values

    def run(self, inputs: List[Union[Image.Image, str]]):
        prompt, input_ids, attention_mask, pixel_values = self.preprocess(inputs)
        with torch.inference_mode():
            # Generate using the model (device_map handles parallelism)
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **self.gen_kwargs
            )
        output = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = {
            "prompt": prompt,
            "output": output,
            "prompt_tokens": input_ids.shape[1],
            "total_tokens": output_ids.shape[1]
        }
        return response

if __name__ == '__main__':
    runner_args = RunnerArguments(model_path='<model_path>')
    runner = OvisRunner(runner_args)
    image = Image.open('<image_path>')
    text = '<prompt>'
    response = runner.run([image, text])
    print(response['output'])
