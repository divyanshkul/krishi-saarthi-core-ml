import os
import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Idefics3ForConditionalGeneration
)
from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionEmbeddings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class VLLMService:
    """
    Vision-Language Model Service for Krishi Saarthi App
    Handles agricultural image analysis using fine-tuned SmolVLM
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.is_loaded = False
        # Use relative path from current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.adapter_path = os.path.join(current_dir, '..', '..', 'ml-models', 'VLLM', 'model_weights')
        self.base_model_id = "HuggingFaceTB/SmolVLM-Base"
    
    def patch_idefics3_vision_embeddings(self):
        """
        Apply the same patch as in training to fix device mismatch issues
        """
        original_forward = Idefics3VisionEmbeddings.forward
        
        def fixed_forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
            batch_size, _, max_im_h, max_im_w = pixel_values.shape

            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
            
            boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
            boundaries = boundaries.to(embeddings.device)
            
            position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)
            position_ids = position_ids.to(embeddings.device)

            for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                p_attn_mask = p_attn_mask.to(embeddings.device)
                
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()
                
                nb_patches_h = nb_patches_h.to(dtype=torch.float32, device=embeddings.device)
                nb_patches_w = nb_patches_w.to(dtype=torch.float32, device=embeddings.device)

                h_indices = torch.arange(nb_patches_h.item(), device=embeddings.device, dtype=torch.float32)
                w_indices = torch.arange(nb_patches_w.item(), device=embeddings.device, dtype=torch.float32)

                fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
                fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
                
                mask_flat = p_attn_mask.view(-1)
                position_ids[batch_idx][mask_flat] = pos_ids.to(position_ids.dtype)

            position_ids = position_ids.to(self.position_embedding.weight.device)
            embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings
        
        Idefics3VisionEmbeddings.forward = fixed_forward
        logger.info("✅ Idefics3VisionEmbeddings patched successfully!")
    
    async def load_model(self):
        """
        Load the fine-tuned SmolVLM model
        """
        if self.is_loaded:
            return
        
        try:
            logger.info("Loading VLLM service for Krishi Saarthi...")
            
            # Apply the vision embeddings patch
            self.patch_idefics3_vision_embeddings()
            
            # Load processor
            logger.info(f"Loading processor from {self.base_model_id}...")
            self.processor = AutoProcessor.from_pretrained(self.base_model_id)
            
            # Validate adapter path
            adapter_path = os.path.abspath(self.adapter_path)
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
            
            config_file = os.path.join(adapter_path, "adapter_config.json")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"adapter_config.json not found at: {config_file}")
            
            # Load model with quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load base model
            base_model = Idefics3ForConditionalGeneration.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            # Load PEFT model (LoRA adapters)
            self.model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
            self.model.eval()
            
            self.is_loaded = True
            logger.info("✅ VLLM service loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load VLLM service: {e}")
            raise
    
    async def generate_response(self, image: Image.Image, question: str) -> str:
        """
        Generate response for agricultural image with question
        
        Args:
            image (PIL.Image): Input image
            question (str): Question about the image
            
        Returns:
            str: AI-generated response
        """
        if not self.is_loaded:
            await self.load_model()
        
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare conversation format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process inputs
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
            
            # Move to device
            device = next(self.model.parameters()).device
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode response
            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )[0]
            
            response = generated_text.strip()
            logger.info(f"Generated response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error during image analysis: {e}")
            raise
    
    def get_status(self) -> dict:
        """
        Get service status
        """
        return {
            "service": "VLLM",
            "loaded": self.is_loaded,
            "model_path": self.adapter_path,
            "base_model": self.base_model_id
        }

# Global service instance
vllm_service = VLLMService()
