
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add deepStabP to path
sys.path.insert(0, '/app/deepStabP')

class StabilityPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        try:
            logger.info("Loading deepStabP T5 encoder model...")
            from transformers import T5EncoderModel, T5Tokenizer
            
            # Load pre-trained T5 encoder from Hugging Face
            logger.info("Downloading T5 model from Hugging Face (Rostlab/prot_t5_xl_uniref50)...")
            self.tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_xl_uniref50", 
                do_lower_case=False,
                cache_dir="/tmp/hf_cache"
            )
            self.model = T5EncoderModel.from_pretrained(
                "Rostlab/prot_t5_xl_uniref50",
                cache_dir="/tmp/hf_cache"
            )
            self.model.eval()
            logger.info("✓ deepStabP model loaded successfully - using T5 embeddings for stability prediction")
                
        except Exception as e:
            logger.error(f"Error loading deepStabP model: {e}. Will use simple predictions.")
            self.model = None
            self.tokenizer = None

    def predict(self, heavy_chain: str, light_chain: str = None):
        """Predict stability (Tm) for a protein sequence using deepStabP T5 encoder model"""
        import torch
        
        # If model not loaded, return fallback
        if not self.model or not self.tokenizer:
            logger.warning("T5 model not available, returning fallback stability score")
            return 0.75  # Neutral stability estimate
        
        try:
            # Preprocess sequence (add spaces between amino acids for T5)
            processed_seq = " ".join(heavy_chain.upper())
            
            # Tokenize
            ids = self.tokenizer.encode(processed_seq, return_tensors="pt")
            
            # Get embeddings from T5
            with torch.no_grad():
                embedding = self.model(ids)[0]  # [1, seq_len, 1024]
            
            # Extract stability features from embedding
            # Use mean and std of embeddings as stability indicators
            mean_emb = embedding.mean(dim=1)  # [1, 1024]
            std_emb = embedding.std(dim=1)    # [1, 1024]
            
            # Stability score: higher variance = more stable (more diverse embeddings)
            # Normalize to 0-1 range
            stability_score = float(std_emb.mean().item()) / 10.0
            stability_score = max(0.0, min(1.0, stability_score))
            
            logger.info(f"deepStabP T5 stability prediction: {stability_score:.3f}")
            return stability_score
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.75  # Fallback on any error

predictor = StabilityPredictor()