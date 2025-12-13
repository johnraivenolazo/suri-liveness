"""Preprocessing utilities for face images."""

import cv2
import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def preprocess_face(face_img, size=224):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    h, w = face_rgb.shape[:2]
    max_dim = max(h, w)
    delta_w = max_dim - w
    delta_h = max_dim - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(
            face_rgb, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
    else:
        img = face_rgb
    if img.shape[0] != size or img.shape[1] != size:
        interp = cv2.INTER_LANCZOS4 if img.shape[0] < size else cv2.INTER_AREA
        img = cv2.resize(img, (size, size), interpolation=interp)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img = (img - mean) / std
    return np.expand_dims(img, axis=0)
