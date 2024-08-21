import logging
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

logger = logging.getLogger(__name__)

prompts = "a person."

device = "cuda"

# logger.info(f"Device: {torch_device}")
# torch.set_grad_enabled(False)

def load_model(model_name):
    try:
        
        
        # Load the pre-trained BLIP model for image-to-text captioning
        processor = AutoProcessor.from_pretrained(model_name)
        model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(device)
        # model = model.to(torch_device)
        logger.info(f"Loaded model: {model_name}")
        return model, processor
    except Exception as e:
        logger.exception(f"Error loading model: {model_name}")
        raise e


def generate_bboxes(model, processor, image, text):
    try:
        #inputs = processor(text=text, images=[image] * len(text), padding="max_length", return_tensors="pt")

        if text is not None:
            # use text prompt supplied by user
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        else:
            # default to person prompt
            inputs = processor(images=image, text=prompts, return_tensors="pt").to(device)

        with torch.no_grad():    
            # predict
            outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.image_processor.post_process_object_detection(
            outputs, threshold=0.35, target_sizes=target_sizes
        )[0]

        logger.info(f"Got some objects")

        result = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):

            box = [round(i, 1) for i in box.tolist()]
            result.append({'box':box, 'label':label.item(), 'confidence':round(score.item(),2)})
            #logger.info(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")

        return result
    
    except Exception as e:
        logger.exception("Error during object detection")
        raise e
