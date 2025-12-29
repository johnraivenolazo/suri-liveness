# Limitations & Technical Notes

Anti-spoofing is texture-sensitive, so input quality directly affects output reliability. Observations from testing and deployment:

## 1. Environmental Constraints

* **Lighting matters:** The model analyzes fine textures (Fourier Transform patterns). In extreme low-light or harsh backlighting, these textures get lost in noise or silhouettes, which can cause misclassification.
* **What works well:** Evenly lit faces perform best. Setups where the camera faces a bright window or strong backlight tend to struggle.

## 2. Input & Preprocessing Requirements

The model was trained on a specific preprocessing pipeline, so it expects inputs that match that pipeline.

* **The 1.5x padding:** The preprocessing uses 1.5x padding (`--bbox_expansion_factor 1.5`) when cropping faces. Tight crops (Just eyes/Forehead) remove the context needed to tell a real head apart from a flat screen or paper. The padding provides the model enough "head space" to see the 3D structure.
* **Resolution:** Input gets resized to 128×128, but the source face should be at least 64×64. If you upscale a tiny, blurry face, you'll lose the spoofing artifacts (like screen pixels) that the model looks for.

## 3. Pose & Occlusion

* **Angles:** Works best with frontal views (±30° yaw/pitch). Profile views or extreme "looking down" angles drop performance noticeably.
* **Obstructions:** Heavy occlusions—masks, hands covering the face, or large-frame glasses with thick reflections—can mess with the texture-based feature extraction.

## 4. Known Edge Cases

* **Attack types:** Handles printed photos and digital screens well. The model hasn't been explicitly trained for 3D silicone masks or high-end prosthetic attacks, so those might slip through.
* **Motion blur:** Fast movement during capture can "smear" the textures. For video streams, using a temporal filter helps (e.g., require 3-5 consecutive "Real" frames before accepting) to avoid false positives from a single blurry frame.

## 5. Security Tuning

The default threshold is balanced for general use (FPR < 2%). Security is a trade-off:

* **High-security:** If you can't afford any spoofs getting through, you can bump the threshold (e.g., 0.5 → 0.8). This will also increase false rejects for real users.
* **High-convenience:** If you want a smoother experience and can tolerate minor risks, lower thresholds should work fine.

## Implementation Example

The 1.5x padding is handled as follows when cropping faces:

```python
import cv2
import numpy as np

def crop_face_with_padding(image, bbox, padding_factor=1.5):
    """
    Crop face with proper padding for anti-spoofing model.
    
    Args:
        image: Input image (numpy array)
        bbox: Bounding box as (x, y, w, h)
        padding_factor: Padding multiplier (default: 1.5)
    
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    
    # Calculate center and expanded dimensions
    center_x = x + w / 2
    center_y = y + h / 2
    max_dim = max(w, h)
    new_size = int(max_dim * padding_factor)
    
    # Calculate new bounding box
    x1 = int(center_x - new_size / 2)
    y1 = int(center_y - new_size / 2)
    x2 = x1 + new_size
    y2 = y1 + new_size
    
    # Clamp to image boundaries
    h_img, w_img = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)
    
    # Crop and resize to 128x128
    face_crop = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    
    return face_resized
```

For temporal filtering in video streams:

```python
from collections import deque

class TemporalFilter:
    """Require N consecutive 'real' predictions before accepting."""
    
    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.history = deque(maxlen=required_frames)
    
    def update(self, is_real: bool) -> bool:
        """Update filter and return final decision."""
        self.history.append(is_real)
        
        if len(self.history) < self.required_frames:
            return False  # Not enough frames yet
        
        return all(self.history)  # All must be 'real'
```
