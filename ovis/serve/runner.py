from dataclasses import field, dataclass
from typing import Optional, Union, List
import os
import torch
from PIL import Image
from accelerate import dispatch_model, infer_auto_device_map
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
        self.dtype = torch.bfloat16  # Use float16 for lower VRAM if needed

        # Use RAM-backed tmpfs for offload_dir
        self.offload_dir = "/dev/shm/offload"
        os.makedirs(self.offload_dir, exist_ok=True)

        # Step 1: Load the model on CPU to compute the device_map
        temp_model = Ovis.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

        # Step 2: Compute device_map with CPU/RAM offloading
        device_map = self._get_balanced_device_map(temp_model)

        # Step 3: Load the model with the computed device_map
        del temp_model  # Free up memory
        self.model = Ovis.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

        # Step 4: Dispatch the model with offload_dir (RAM-backed)
        dispatch_model(
            self.model,
            device_map=device_map,
            offload_dir=self.offload_dir,  # Use RAM-backed directory
            main_device=torch.cuda.current_device(),
        )

        self.model.eval()
        # ... rest of the code remains the same ...

    def _get_balanced_device_map(self, model):
        num_gpus = torch.cuda.device_count()
        max_memory = {}

        # Assign 90% of each GPU's memory
        for i in range(num_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = f"{int(total_mem * 0.9 / 1e9)}GB"

        # Use available RAM (e.g., 100GB allocated via tmpfs)
        max_memory["cpu"] = "100GB"  # Adjust based on your tmpfs size

        # Specify modules that cannot be split
        no_split = ["VisualTransformerEmbedding", "vte", "visual_tokenizer"]

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
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

        # Move tensors to device1 (GPU 0)
        input_ids = input_ids.to(self.device1)
        attention_mask = attention_mask.to(self.device1)
        pixel_values = [pv.to(self.device1, dtype=self.dtype) for pv in pixel_values]
    
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
