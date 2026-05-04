import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from logger_utils import logger

class LLMTranslationPipeline:
    def __init__(self, model_id="unsloth/Llama-3.2-3B-Instruct-bnb-4bit"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading local LLM for translation ({model_id}) on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

    def translate(self, text, source_lang, target_lang):
        """Translates text using the LLM with a specific system prompt."""
        prompt = f"""Translate the following text from {source_lang} to {target_lang}. 
Keep the tone and emotion of the original. Only output the translation.

Text: {text}
Translation:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0
            )
            
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the translation after the prompt
        translation = full_text.split("Translation:")[-1].strip()
        return translation

if __name__ == "__main__":
    pass
