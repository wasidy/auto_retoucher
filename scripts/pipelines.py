import gc
import torch
import numpy as np
import cv2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerImageProcessor
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from typing import List


class ClipSegmentation():
    ''' CLIP Segmentation class '''

    def __init__(self, model_name, processor_name):
        ''' model_name and processor_name must be from HuggingFace hub '''
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        self.processor = CLIPSegProcessor.from_pretrained(processor_name)

    def predict(self, image, prompt, threshold=127, kernel_size=8, sigma=11):
        ''' Segmentation with CLIPSegProcessor. Supported only one prompt.
            Outputs is grayscaled binary mask.
            Threshold if sensivity of detection, higher value more stronger.
            Kernel size used for morphological opening (see openCV doc).
            Sigma is value for gaussian blur for smoothing borders. Only odd value.'''

        if image is None:
            return None

        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        height, width = image.shape[0:2]

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = outputs.logits

        mask = np.array(np.clip(torch.sigmoid(preds), 0, 1)*255, dtype=np.uint8)[:, :, np.newaxis]
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.resize(mask, (width, height))
        mask = cv2.GaussianBlur(mask, (sigma, sigma), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask


class Mask2FormerSegmentation():
    ''' Mask2Former Segmentation '''

    def __init__(self, model_name, processor_name):
        ''' model_name and processor_name must be from HuggingFace hub '''
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.processor = Mask2FormerImageProcessor.from_pretrained(processor_name)

    def predict(self, image, label_id=0):
        ''' Generates mask from image with Mask2Former panoptic segmentation
            Masks with selected labels will be fused '''

        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)

        inputs = self.processor(image, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)
        pred = self.processor.post_process_panoptic_segmentation(outputs,
                                                                 label_ids_to_fuse=[label_id],
                                                                 target_sizes=[image.shape[0:2]])[0]
        idx: int = None
        for key in pred['segments_info']:
            if key['label_id'] == label_id:
                idx = key['id']
        if idx is not None:
            temp_mask = np.array(pred['segmentation'], dtype=np.uint8)
            mask = np.array([temp_mask == idx], dtype=np.uint8)[0, ::]*255
        else:
            return None
        return mask


class SDXLPipeLine():
    ''' SDXL Pipeline for Img2Img generation '''

    def __init__(self, checkpoint_file: str = None):
        ''' SDXL checkpoint file for loading at start '''
        self.pipe: StableDiffusionXLImg2ImgPipeline = None
        self.checkpoint = checkpoint_file
        self.stop_button_pressed = False

        if self.checkpoint is not None:
            self.load_pipe(self.checkpoint)

    def load_pipe(self, checkpoint_file):
        ''' Load SDXL checkpoint to pipe '''

        if self.pipe is not None:
            self.delete_pipe()

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                        checkpoint_file,
                        torch_dtype=torch.float16,
                        use_safetensors=True)

        self.pipe = self.pipe.to('cuda')
        self.pipe.enable_xformers_memory_efficient_attention()
        self.checkpoint = checkpoint_file

    def delete_pipe(self):
        ''' Delete loaded pipe from memory '''
        if self.pipe is not None:
            del self.pipe
            gc.collect()
            torch.cuda.empty_cache()
            self.pipe: StableDiffusionXLImg2ImgPipeline = None

    def generate_single_image(self, image, strength, prompt='',
                              negative_prompt='', steps=50) -> Image.Image:
        ''' image - PIL or numpy, strength (0...100) '''

        if isinstance(image, np.ndarray):
            image = Image.fomarray(image)
        generated_image = np.array(self.pipe(prompt=prompt, negative_prompt=negative_prompt,
                                             image=image,
                                             num_inference_steps=steps,
                                             strength=strength/100.0).images[0])
        return generated_image

    def generate_batch_images(self, images: List, strength, prompt='',
                              negative_prompt='', steps=50, batch_size=1) -> List[Image.Image]:
        ''' Generating images from list PIL or numpy '''

        generated_images = []
        strength = strength/100.0

        if images:
            next_elem = True
            iter_img = iter(images)
        else:
            return None

        while next_elem:
            if self.stop_button_pressed:
                self.stop_button_pressed = False
                return None
            batch_images = []

            for i in range(batch_size):
                try:
                    temp_image = next(iter_img)
                    if isinstance(temp_image, np.ndarray):
                        temp_image = Image.fromarray(temp_image)
                    batch_images.append(temp_image)
                except StopIteration:
                    next_elem = False
                    break

            if batch_images:
                num_per_prompt = len(batch_images)
                outputs = self.pipe(prompt=prompt, negative_prompt=negative_prompt,
                                    image=batch_images, num_images_per_prompt=num_per_prompt,
                                    num_inference_steps=steps, strength=strength).images
                for im in outputs:
                    generated_images.append(np.array(im))
        return generated_images
