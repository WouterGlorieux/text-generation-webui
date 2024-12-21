import time
from typing import List, Optional, Tuple
import math
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from PIL import Image

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.logging_colors import logger
from modules.text_generation import encode

class GatedMLP(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim)
        self.w2 = torch.nn.Linear(dim, hidden_dim)
        self.w3 = torch.nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))

class PixtralViTBlock(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            kdim=dim,
            vdim=dim,
            batch_first=True
        )
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = GatedMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attention_mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class PixtralViT(torch.nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        n_layers: int = 24,
        head_dim: int = 64,
        hidden_dim: int = 4096,
        n_heads: int = 16,
        patch_size: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = torch.nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Transformer blocks
        self.blocks = torch.nn.ModuleList([
            PixtralViTBlock(dim, n_heads, head_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.norm = torch.nn.LayerNorm(dim)

    def _create_2d_rope_embeddings(self, height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Following equation (1) in the paper for ROPE-2D implementation
        theta = 10000.0
        dim = self.dim // 4  # Split dim for height and width components
        
        freq_h = torch.arange(dim, device=device) / dim
        freq_h = 1.0 / (theta ** freq_h)
        
        freq_w = torch.arange(dim, device=device) / dim
        freq_w = 1.0 / (theta ** freq_w)
        
        pos_h = torch.arange(height, device=device).unsqueeze(1) * freq_h.unsqueeze(0)
        pos_w = torch.arange(width, device=device).unsqueeze(1) * freq_w.unsqueeze(0)
        
        return pos_h, pos_w

    def _apply_rope_2d(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        pos_h, pos_w = self._create_2d_rope_embeddings(height, width, x.device)
        
        # Apply rotary embeddings
        x_rope = x.clone()
        for h in range(height):
            for w in range(width):
                idx = h * width + w
                if idx < x.size(1):  # Check sequence length
                    # Apply height rotation
                    x_rope[:, idx, ::4] = x[:, idx, ::4] * torch.cos(pos_h[h]) - x[:, idx, 1::4] * torch.sin(pos_h[h])
                    x_rope[:, idx, 1::4] = x[:, idx, ::4] * torch.sin(pos_h[h]) + x[:, idx, 1::4] * torch.cos(pos_h[h])
                    
                    # Apply width rotation
                    x_rope[:, idx, 2::4] = x[:, idx, 2::4] * torch.cos(pos_w[w]) - x[:, idx, 3::4] * torch.sin(pos_w[w])
                    x_rope[:, idx, 3::4] = x[:, idx, 2::4] * torch.sin(pos_w[w]) + x[:, idx, 3::4] * torch.cos(pos_w[w])
        
        return x_rope

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Create block-diagonal attention mask for multiple images
        n_patches = (height // self.patch_size) * (width // self.patch_size)
        attention_mask = torch.ones((n_patches, n_patches), device=x.device) * float('-inf')
        # Fill diagonal blocks with 0 to allow attention within each image
        for i in range(0, n_patches, width // self.patch_size):
            end = min(i + width // self.patch_size, n_patches)
            attention_mask[i:end, i:end] = 0
        
        # Apply ROPE-2D
        x = self._apply_rope_2d(x, height // self.patch_size, width // self.patch_size)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        return self.norm(x)

class Pixtral_12B_Pipeline(AbstractMultimodalPipeline):
    PIXTRAL_VIT_REPO = "Pixtral/Pixtral-ViT"  # Update when official repo is available

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.vision_device = self._get_device("vision_device", params)
        self.vision_dtype = self._get_dtype("vision_bits", params)
        self.projector_device = self._get_device("projector_device", params)
        self.projector_dtype = self._get_dtype("projector_bits", params)
        self.vision_model, self.mm_projector = self._load_models()

    def _load_models(self):
        try:
            # Load vision model
            logger.info("Loading Pixtral vision encoder...")
            vision_path = hf_hub_download(self.PIXTRAL_VIT_REPO, "pytorch_model.bin")
            vision_config = AutoConfig.from_pretrained(self.PIXTRAL_VIT_REPO)
            vision_model = PixtralViT(
                dim=vision_config.hidden_size,
                n_layers=vision_config.num_hidden_layers,
                head_dim=vision_config.head_dim,
                hidden_dim=vision_config.intermediate_size,
                n_heads=vision_config.num_attention_heads,
                patch_size=vision_config.patch_size
            )
            state_dict = torch.load(vision_path, map_location='cpu')
            vision_model.load_state_dict(state_dict)
            vision_model.to(self.vision_device, dtype=self.vision_dtype)
            vision_model.eval()
            
            # Load projector
            logger.info("Loading Pixtral projector...")
            projector = self.build_mm_projector()
            projector_path = hf_hub_download(self.pixtral_projector_repo(), self.pixtral_projector_filename())
            state_dict = torch.load(projector_path, map_location='cpu')
            projector.load_state_dict(state_dict)
            projector.to(self.projector_device, dtype=self.projector_dtype)
            projector.eval()
            
            return vision_model, projector
            
        except Exception as e:
            logger.error(f"Error loading Pixtral models: {str(e)}")
            raise

    def build_mm_projector(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),  # From Table 1: vision dim -> hidden_dim
            torch.nn.GELU(),
            torch.nn.Linear(4096, 5120)   # From Table 1: hidden_dim -> decoder dim
        )

    @staticmethod
    def name() -> str:
        return "pixtral-12b"

    @staticmethod
    def image_start() -> str:
        return "[IMG START]"

    @staticmethod
    def image_break() -> str:
        return "[IMG BREAK]"

    @staticmethod
    def image_end() -> str:
        return "[IMG END]"

    def num_image_embeds(self, image_size: tuple) -> int:
        # Calculate patches based on image dimensions
        height_patches = (image_size[0] + self.vision_model.patch_size - 1) // self.vision_model.patch_size
        width_patches = (image_size[1] + self.vision_model.patch_size - 1) // self.vision_model.patch_size
        return height_patches * width_patches

    @staticmethod
    def placeholder_token_id() -> int:
        return 32001

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        image_tensors = []
        for image in images:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
            image_tensor = image_tensor / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            # Process with vision model
            with torch.no_grad():
                image_tensor = image_tensor.to(self.vision_device, dtype=self.vision_dtype)
                height, width = image_tensor.shape[-2:]
                image_features = self.vision_model(image_tensor, height, width)
                image_features = self.mm_projector(image_features)
                
            image_tensors.append(image_features)
        
        return torch.cat(image_tensors, dim=1)

    def pixtral_projector_filename(self) -> str:
        return "mm_projector.bin"

    def pixtral_projector_repo(self) -> str:
        return "Pixtral/Pixtral-12B-VL"  # Update when official repo is available

    @staticmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        for attr in ['', 'model', 'model.model', 'model.model.model']:
            tmp = getattr(shared.model, attr, None) if attr != '' else shared.model
            if tmp is not None and hasattr(tmp, 'embed_tokens'):
                func = tmp.embed_tokens
                break
        else:
            raise ValueError('The embed_tokens method has not been found for this loader.')

        return func(input_ids).to(shared.model.device, dtype=shared.model.dtype)

    def placeholder_embeddings(self, num_patches: int) -> torch.Tensor:
        # Generate placeholder embeddings based on the actual number of patches
        return self.embed_tokens(encode("<image_patch>" * num_patches, add_bos_token=False)[0])
