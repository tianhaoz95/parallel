import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from logger_utils import logger

class VLMPrompter:
    def __init__(self, model_id="models/moondream2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading local VLM for contextual prompting ({model_id}) on {self.device}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def describe_frame(self, frame_np):
        """Generates a detailed description of the frame context."""
        image = Image.fromarray(frame_np)
        
        # Moondream specific prompt
        prompt = "Describe the lighting, environment, and background of this scene in detail for an image generation task."
        
        with torch.no_grad():
            # Encoded image features
            image_embeds = self.model.encode_image(image)
            # Answer generation
            answer = self.model.answer_question(image_embeds, prompt, self.tokenizer)
            
        logger.info(f"VLM Frame Context: {answer}")
        return answer

if __name__ == "__main__":
    pass
