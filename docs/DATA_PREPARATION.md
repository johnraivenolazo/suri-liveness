# Data Preparation

This documentation explains the preprocessing decisions in `scripts/prepare_data.py`.

---

## The Pipeline

For each image:

1. Read the bounding box from `image_BB.txt`
2. Expand the box by 1.5× (configurable)
3. Make it square (use the longer side)
4. Pad if the crop goes outside image bounds
5. Resize to 128×128
6. Save


---

## Why Expand the Bounding Box?

Face detectors give you tight bounding boxes. Just the face, sometimes cutting off the forehead or chin. For anti-spoofing, context is needed:

- **Skin texture around the face** helps distinguish real skin from printed paper
- **Hair and ears** often show printing artifacts in spoof attacks
- **The boundary between face and background** is where you see edges of printed photos or phone bezels

The default expansion is **1.5×**, meaning the crop is taken 50% larger than the detected face on each side. This provides the needed context without including too much irrelevant background.

![Bbox](../assets/docs/bbox.png)

---

## Why Square Crops?

The model expects 128×128 square inputs. Cropping a rectangle and stretching it distorts facial proportions. Distortion patterns differ between real faces and spoofs. The goal is to avoid the model learning "stretched faces are spoofs."

The approach:
1. Take the longer side of the bounding box
2. Make a square crop centered on the face
3. Resize that square to 128×128

No distortion, consistent aspect ratio.

---

## The Padding Problem

Sometimes the expanded crop goes outside the image bounds. This happens when:
- The face is near the edge of the frame
- The expansion factor is large
- The original image is tightly cropped

Three options are available:
1. **Skip the image**: loses training data
2. **Pad with black**: creates harsh artificial edges
3. **Pad with reflected pixels**: extends the image naturally

Option 3 is used: `cv2.BORDER_REFLECT_101`.

### Why BORDER_REFLECT_101?

There are several reflection modes:

| Mode | What it does |
|:-----|:-------------|
| `BORDER_REFLECT` | `abcdef` → `fedcba│abcdef│fedcba` |
| `BORDER_REFLECT_101` | `abcdef` → `gfedcb│abcdef│edcbag` |
| `BORDER_REPLICATE` | `abcdef` → `aaaaaa│abcdef│ffffff` |

`BORDER_REFLECT_101` (also called "reflect without edge duplication") gives the smoothest result. The edge pixel isn't repeated, so you don't get a visible seam.

In practice:
- **Black padding** creates edges the model might learn as "spoof indicators" (they're not)
- **Replicate** creates visible bands of repeated color
- **Reflect 101** looks like a natural extension of the image

Avoids introducing artifacts that the model might wrongly learn as spoof indicators.

![Padding comparison](../assets/docs/padding_comparison.png)

---

## Resizing: LANCZOS vs AREA

```python
interp = cv2.INTER_LANCZOS4 if crop_size < target_size else cv2.INTER_AREA
```

### Upscaling (small crop → 128×128)

When the face is small and scaling up, **LANCZOS4** is used.

LANCZOS is a high-quality interpolation that preserves edges and fine details. It uses a windowed sinc function to sample surrounding pixels. Sharp results without the blocky look of nearest-neighbor or the blur of bilinear.

Small faces often come from subjects far from the camera. Better to preserve texture than blur it away.

![Upscale comparison](../assets/docs/upscale_comparison.png)

### Downscaling (large crop → 128×128)

When the face is large and scaling down, **AREA** is used.

AREA interpolation (also called "pixel area relation") averages the pixels that map to each output pixel. Proper downsampling that considers all source pixels, not just a few sample points.

Naive methods (bilinear, bicubic) can miss fine details or create aliasing. AREA gives a true average, keeping texture information intact.

![Downscale comparison](../assets/docs/downscale_comparison.png)

### The Quick Version

| Situation | Method | Why |
|:----------|:-------|:----|
| Upscaling | LANCZOS4 | Sharp edges, preserves detail |
| Downscaling | AREA | Proper averaging, no aliasing |

---

## What Gets Saved

After processing:

```
{crop_dir}/
├── train/
│   └── {same structure as original dataset}
├── test/
│   └── {same structure as original dataset}
└── metas/
    └── labels/
        ├── train_label.json
        └── test_label.json
```

The script preserves the original folder structure. If your source has `train/subject_001/image.jpg`, the output will have the same path under `{crop_dir}`. The label JSON files are copied to `metas/labels/` so the training script knows where to find them.

---

## TL;DR

| Decision | Choice | Reason |
|:---------|:-------|:-------|
| Bbox expansion | 1.5× | Include context (hair, ears, skin texture) |
| Crop shape | Square | No distortion, consistent input |
| Padding | REFLECT_101 | Natural extension, no artificial edges |
| Upscale | LANCZOS4 | Sharp, preserves detail |
| Downscale | AREA | Proper averaging, no aliasing |


