"""
Module 2: Expert Grounding (Explicit Semantic Concepts)
Uses DeepFace to extract interpretable semantic attributes:
- Gender, Age, Emotion, Race
- Provides explicit semantic grounding for causal discovery

This module extracts human-interpretable concepts that can be used
as explicit variables in the causal graph.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # <--- This fixes the DeepFace/Keras crash


logger = logging.getLogger(__name__)


class DeepFaceSemanticExtractor(nn.Module):
    """
    Extract explicit semantic concepts using DeepFace
    
    DeepFace provides:
    - Age estimation
    - Gender classification  
    - Emotion recognition (angry, disgust, fear, happy, sad, surprise, neutral)
    - Race classification (asian, indian, black, white, middle eastern, latino hispanic)
    - Face verification
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config.get('semantic_grounding', {})
        self.enabled = self.config.get('enabled', True)
        
        # DeepFace will be imported lazily to avoid dependency issues
        self.deepface = None
        self.deepface_backends = ['opencv', 'ssd', 'mtcnn', 'retinaface']
        self.detector_backend = self.config.get('detector_backend', 'retinaface')
        
        # Which attributes to extract
        self.extract_age = self.config.get('extract_age', True)
        self.extract_gender = self.config.get('extract_gender', True)
        self.extract_emotion = self.config.get('extract_emotion', True)
        self.extract_race = self.config.get('extract_race', False)  # Optional
        
        # Emotion mapping (7 emotions)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Gender mapping
        self.genders = ['Man', 'Woman']
        
        # Race mapping (optional)
        self.races = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
        
        # Output dimensions
        self.age_dim = 1  # Normalized age
        self.gender_dim = 2  # One-hot
        self.emotion_dim = 7  # Emotion probabilities
        self.race_dim = 6 if self.extract_race else 0
        
        self.total_dim = self.age_dim + self.gender_dim + self.emotion_dim + self.race_dim
        
        # For batch processing, we'll cache face detections
        self.use_cache = True
        
        logger.info(f"DeepFace Semantic Extractor initialized (total_dim={self.total_dim})")
    
    def _lazy_load_deepface(self):
        """Lazy load DeepFace library"""
        if self.deepface is None:
            try:
                from deepface import DeepFace
                self.deepface = DeepFace
                logger.info("DeepFace loaded successfully")
            except ImportError:
                logger.error(
                    "DeepFace not installed. Install with: pip install deepface"
                )
                raise ImportError(
                    "DeepFace is required for semantic concept extraction. "
                    "Install with: pip install deepface"
                )
    
    def analyze_face(self, image_array: np.ndarray, enforce_detection: bool = False) -> Dict:
        """
        Analyze a single face image using DeepFace
        
        Args:
            image_array: RGB image as numpy array (H, W, 3)
            enforce_detection: If True, raise error if no face detected
        
        Returns:
            Dictionary with age, gender, emotion, race
        """
        self._lazy_load_deepface()
        
        try:
            # DeepFace.analyze returns age, gender, emotion, race
            actions = []
            if self.extract_age:
                actions.append('age')
            if self.extract_gender:
                actions.append('gender')
            if self.extract_emotion:
                actions.append('emotion')
            if self.extract_race:
                actions.append('race')
            
            result = self.deepface.analyze(
                img_path=image_array,
                actions=actions,
                enforce_detection=enforce_detection,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                result = result[0]  # Take first face
            
            return result
            
        except Exception as e:
            logger.warning(f"DeepFace analysis failed: {e}")
            # Return default values
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when face detection fails"""
        default = {}
        
        if self.extract_age:
            default['age'] = 30  # Default age
        
        if self.extract_gender:
            default['gender'] = {'Man': 50, 'Woman': 50}  # Neutral
        
        if self.extract_emotion:
            # Neutral emotion
            default['emotion'] = {
                'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
                'sad': 0, 'surprise': 0, 'neutral': 100
            }
        
        if self.extract_race:
            # Equal probability for all races
            default['race'] = {race: 100/6 for race in self.races}
        
        return default
    
    def parse_analysis_to_vector(self, analysis: Dict) -> torch.Tensor:
        """
        Convert DeepFace analysis to a feature vector
        
        Args:
            analysis: Output from DeepFace.analyze()
        
        Returns:
            Tensor of shape (total_dim,)
        """
        features = []
        
        # Age (normalized to 0-1)
        if self.extract_age:
            age = analysis.get('age', 30)
            age_normalized = age / 100.0  # Normalize to [0, 1]
            features.append(age_normalized)
        
        # Gender (one-hot or probabilities)
        if self.extract_gender:
            gender_dict = analysis.get('gender', {'Man': 50, 'Woman': 50})
            if isinstance(gender_dict, dict):
                # Convert to probabilities
                man_prob = gender_dict.get('Man', 50) / 100.0
                woman_prob = gender_dict.get('Woman', 50) / 100.0
            else:
                # Dominant gender string
                man_prob = 1.0 if gender_dict == 'Man' else 0.0
                woman_prob = 1.0 if gender_dict == 'Woman' else 0.0
            
            features.extend([man_prob, woman_prob])
        
        # Emotion (7-dimensional probability vector)
        if self.extract_emotion:
            emotion_dict = analysis.get('emotion', {})
            emotion_probs = []
            for emotion in self.emotions:
                prob = emotion_dict.get(emotion, 100/7) / 100.0  # Normalize to [0, 1]
                emotion_probs.append(prob)
            features.extend(emotion_probs)
        
        # Race (optional)
        if self.extract_race:
            race_dict = analysis.get('race', {})
            race_probs = []
            for race in self.races:
                prob = race_dict.get(race, 100/6) / 100.0
                race_probs.append(prob)
            features.extend(race_probs)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract semantic concepts from a batch of images
        
        Args:
            images: (B, C, H, W) or (B, N, C, H, W) batch of images
                    Values should be in [0, 1] range (normalized)
        
        Returns:
            semantic_features: (B, total_dim) or (B, N, total_dim)
        """
        if not self.enabled:
            # Return zeros if disabled
            if len(images.shape) == 4:
                B = images.size(0)
                return torch.zeros(B, self.total_dim, device=images.device)
            else:
                B, N = images.shape[:2]
                return torch.zeros(B, N, self.total_dim, device=images.device)
        
        # Check input shape
        if len(images.shape) == 5:
            # (B, N, C, H, W) - multiple images per sample
            B, N, C, H, W = images.shape
            images = images.reshape(B * N, C, H, W)
            reshape_output = True
        else:
            # (B, C, H, W)
            B = images.size(0)
            reshape_output = False
        
        # Convert to numpy (DeepFace expects RGB numpy arrays)
        # Denormalize from [0,1] to [0,255]
        images_np = (images * 255).byte().cpu().numpy()
        
        # Process each image
        batch_features = []
        for i in range(images_np.shape[0]):
            # Convert from (C, H, W) to (H, W, C)
            img = np.transpose(images_np[i], (1, 2, 0))
            
            # Analyze face
            analysis = self.analyze_face(img, enforce_detection=False)
            
            # Convert to vector
            features = self.parse_analysis_to_vector(analysis)
            batch_features.append(features)
        
        # Stack to batch
        semantic_features = torch.stack(batch_features, dim=0).to(images.device)
        
        # Reshape if needed
        if reshape_output:
            semantic_features = semantic_features.reshape(B, N, -1)
        
        return semantic_features
    
    def get_concept_names(self) -> List[str]:
        """Get names of all semantic concepts"""
        names = []
        
        if self.extract_age:
            names.append('age')
        
        if self.extract_gender:
            names.extend(['gender_man', 'gender_woman'])
        
        if self.extract_emotion:
            names.extend([f'emotion_{e}' for e in self.emotions])
        
        if self.extract_race:
            names.extend([f'race_{r.replace(" ", "_")}' for r in self.races])
        
        return names
    
    def interpret_features(self, features: torch.Tensor) -> Dict[str, any]:
        """
        Convert feature vector back to interpretable format
        
        Args:
            features: (total_dim,) tensor
        
        Returns:
            Dictionary with interpretable values
        """
        features = features.cpu().numpy()
        interpretation = {}
        idx = 0
        
        if self.extract_age:
            age = features[idx] * 100  # Denormalize
            interpretation['age'] = int(age)
            idx += 1
        
        if self.extract_gender:
            man_prob, woman_prob = features[idx:idx+2]
            interpretation['gender'] = 'Man' if man_prob > woman_prob else 'Woman'
            interpretation['gender_confidence'] = max(man_prob, woman_prob)
            idx += 2
        
        if self.extract_emotion:
            emotion_probs = features[idx:idx+7]
            dominant_emotion_idx = np.argmax(emotion_probs)
            interpretation['emotion'] = self.emotions[dominant_emotion_idx]
            interpretation['emotion_confidence'] = emotion_probs[dominant_emotion_idx]
            idx += 7
        
        if self.extract_race:
            race_probs = features[idx:idx+6]
            dominant_race_idx = np.argmax(race_probs)
            interpretation['race'] = self.races[dominant_race_idx]
            interpretation['race_confidence'] = race_probs[dominant_race_idx]
            idx += 6
        
        return interpretation


class SemanticGroundingModule(nn.Module):
    """
    Module 2: Expert Grounding
    Combines implicit features with explicit semantic concepts from DeepFace
    """
    
    def __init__(self, config, feature_dim: int):
        super().__init__()
        
        self.config = config
        self.feature_dim = feature_dim  # Dimension of input features (from fusion)
        
        # DeepFace extractor
        self.deepface_extractor = DeepFaceSemanticExtractor(config)
        self.semantic_dim = self.deepface_extractor.total_dim
        
        # Combine implicit features with explicit concepts
        self.use_fusion = config.get('semantic_grounding', {}).get('fuse_with_features', True)
        
        if self.use_fusion:
            # Project combined features
            combined_dim = feature_dim + self.semantic_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(combined_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256)
            )
            self.output_dim = 256
        else:
            # Just use semantic concepts
            self.output_dim = self.semantic_dim
        
        logger.info(
            f"Semantic Grounding Module initialized: "
            f"semantic_dim={self.semantic_dim}, output_dim={self.output_dim}"
        )
    
    def forward(self, features: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and combine semantic concepts with features
        
        Args:
            features: (B, feature_dim) - features from Module 1
            images: (B, N, C, H, W) - spatial frames for semantic extraction
        
        Returns:
            combined_features: (B, output_dim) - features with semantic grounding
            semantic_concepts: (B, semantic_dim) - explicit semantic concepts
        """
        # Extract semantic concepts from images
        # Average across multiple frames
        B, C, H, W = image.shape
        semantic_concepts = self.deepface_extractor(image)  # (B, N, semantic_dim)
        
        # Average across frames to get video-level concepts
        # semantic_concepts = semantic_per_frame.mean(dim=1)  # (B, semantic_dim)
        
        if self.use_fusion:
            # Combine with implicit features
            combined = torch.cat([features, semantic_concepts], dim=1)
            combined_features = self.fusion_layer(combined)
        else:
            combined_features = semantic_concepts
        
        return combined_features, semantic_concepts
    
    def get_concept_names(self) -> List[str]:
        """Get names of semantic concepts"""
        return self.deepface_extractor.get_concept_names()
    
    def interpret_concepts(self, semantic_concepts: torch.Tensor) -> List[Dict]:
        """
        Interpret semantic concepts for a batch
        
        Args:
            semantic_concepts: (B, semantic_dim)
        
        Returns:
            List of interpretation dicts, one per sample
        """
        interpretations = []
        for i in range(semantic_concepts.size(0)):
            interp = self.deepface_extractor.interpret_features(semantic_concepts[i])
            interpretations.append(interp)
        
        return interpretations


# Example usage for testing
def test_deepface_extractor():
    """Test the DeepFace semantic extractor"""
    
    # Create dummy config
    config = {
        'semantic_grounding': {
            'enabled': True,
            'detector_backend': 'opencv',  # Faster for testing
            'extract_age': True,
            'extract_gender': True,
            'extract_emotion': True,
            'extract_race': False
        }
    }
    
    # Create extractor
    extractor = DeepFaceSemanticExtractor(config)
    
    # Test with dummy image
    dummy_image = torch.rand(4, 3, 224, 224)  # Batch of 4 images
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output dimension: {extractor.total_dim}")
    print(f"Concept names: {extractor.get_concept_names()}")
    
    # Extract features
    features = extractor(dummy_image)
    print(f"Output shape: {features.shape}")
    
    # Interpret first sample
    interpretation = extractor.interpret_features(features[0])
    print(f"Interpretation: {interpretation}")
    
    return extractor


if __name__ == '__main__':
    # Test the module
    test_deepface_extractor()