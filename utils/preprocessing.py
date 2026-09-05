"""
Image preprocessing for the CRNN.

Turns an arbitrary uploaded/captured photo of handwriting into the
fixed-size, normalized single-channel tensor the model expects:
    (1, IMG_HEIGHT, IMG_WIDTH, 1) float32 in [0, 1]

Pipeline: decode -> grayscale -> resize-with-aspect-ratio (fill
height, cap width) -> pad remaining width with white -> scale to
[0, 1].

*** FIX NOTE ***
The previous version of this file additionally ran a bilateral
denoise -> adaptive threshold (hard binarize to pure black/white) ->
morphological close, BEFORE resizing. Most CRNN/CTC handwriting
models (this architecture — 64px height, CNN + BiLSTM + CTC — is a
classic IAM-style setup) are trained on plain anti-aliased grayscale
images, not binarized ones. Feeding a hard-binarized image to a model
trained on smooth grayscale strokes changes the input distribution
enough to degrade or corrupt predictions, independent of the vocab
bug already fixed in app.py.

That block is now OFF by default (`binarize=False`). If your training
notebook/script shows images were thresholded the same way before
training, set `binarize=True` (or tell me the exact steps used and
I'll match them precisely) — matching training exactly matters more
than which specific steps are used.
"""

import cv2
import numpy as np


def _resize_keep_aspect(gray, target_h, target_w):
    """Resize so the image fills the target height, preserving aspect
    ratio, then pad (or crop) width to exactly target_w with a white
    background — the common HTR convention of dark ink on a light
    page. Returns a (target_h, target_w) uint8 array."""
    h, w = gray.shape
    scale = target_h / float(h)
    new_w = min(target_w, max(1, int(round(w * scale))))
    resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)  # white page
    canvas[:, :new_w] = resized
    return canvas


def preprocess_image(image_bytes, img_height=64, img_width=1024, binarize=False):
    """
    image_bytes: raw bytes of an uploaded or captured image (any
        format OpenCV can decode: jpg/png/webp/bmp).
    img_height, img_width: target model input size (from config.py).
    binarize: set True ONLY if your training pipeline hard-thresholded
        images to pure black/white before feeding the CNN. Defaults to
        False — see the FIX NOTE above for why.

    Returns: (model_input, display_image)
        model_input   -> np.ndarray shaped (1, H, W, 1), float32 [0, 1]
        display_image -> np.ndarray (H, W) uint8, the processed image
                          (useful for a "preprocessed preview" panel)
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported or corrupt file.")

    # Grayscale conversion — same weighting cv2 and PIL's "L" mode use,
    # so this matches however training data was likely grayscaled.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if binarize:
        # Only runs if explicitly requested. Off by default — see the
        # FIX NOTE at the top of this file.
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=25, C=15,
        )
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Resize preserving aspect ratio to fill the target height, then
    # pad the remaining width with white — matches standard HTR
    # convention (e.g. IAM-style pipelines).
    canvas = _resize_keep_aspect(gray, img_height, img_width)

    # Normalize to [0, 1] — the standard range for grayscale CRNN
    # inputs (as opposed to [-1, 1], which is more common for
    # ImageNet-style classification backbones).
    normalized = canvas.astype("float32") / 255.0
    model_input = np.expand_dims(normalized, axis=(0, -1))  # (1, H, W, 1)

    return model_input, canvas
