from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
import os
from logging import warning
import streamlit as st

ROOT_DIR = Path('..')

import os
from pathlib import Path
import numpy as np
from IPython.display import display, clear_output
import ipywidgets as widgets
from PIL import Image
from typing import List, Tuple, Optional, Callable, Iterator
import cv2
import numpy as np

BASE_DIR = Path(f'{ROOT_DIR}/data/labeled')

IMAGE_FILE_TYPE = 'png'  


def ensure_dirs(base_dir:Path =BASE_DIR)-> Tuple[Path, Path]:
    pos_dir = Path(base_dir) / "positive"
    neg_dir = Path(base_dir) / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    return pos_dir, neg_dir

def normalize_image(img: np.ndarray) -> np.ndarray:
    mi, ma = float(img.min()), float(img.max())
    if ma > mi:
        norm = (img - mi) / (ma - mi) * 255.0
    else:
        norm = np.zeros_like(img)
    return norm.astype(np.uint8)


def preprocess_image(
    img: np.ndarray,
    size: Tuple[int, int] = (300, 300),
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    1. Normalize raw floats to 0–255 uint8
    2. Convert to single-channel if needed
    3. Resize to `size`
    4. Apply CLAHE
    5. Return float32 image in [0,1]
    """
    # 1. Scale floats → 0–255
    uint8_img = normalize_image(img)
    
    # 2. Ensure single‑channel
    if uint8_img.ndim == 3 and uint8_img.shape[2] > 1:
        uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2GRAY)
    
    # 3. Resize
    resized = cv2.resize(uint8_img, size, interpolation=cv2.INTER_LINEAR)
    
    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(resized)
    
    # 5. Back to uint8 for display
    return clahe_img.astype(np.uint8)


def is_already_labeled(name: str, pos_dir: Path, neg_dir: Path) -> bool:
    return (pos_dir/f"{name}.{IMAGE_FILE_TYPE}").exists() or (neg_dir/f"{name}.{IMAGE_FILE_TYPE}").exists()

def save_image(img: np.ndarray, name: str, target_dir: Path):
    uint8_img = normalize_image(img)
    Image.fromarray(uint8_img).save(target_dir / f"{name}.{IMAGE_FILE_TYPE}")

def get_image(X: np.ndarray, index: Tuple[int, int]) -> np.ndarray:
    first_index, second_index = index
    if first_index >= X.shape[0] or second_index >= X.shape[-1]:
        raise IndexError("Index out of bounds for the image array.")
    return X[first_index, :, :, second_index]

def get_image_name(index: Tuple[int, int]) -> str:
    first_index, second_index = index
    return f"image_{first_index}_channel_{second_index}"

def random_index_iterator(dim0: int, dim1: int) -> Iterator[Tuple[int, int]]:
    indices = [(i, j) for i in range(dim0) for j in range(dim1)]
    random.shuffle(indices)
    for idx in indices:
        yield idx

def main():
    st.title("Image Labeling Tool")

    X =  np.load(f'{ROOT_DIR}/data/proccessed/X.npy')
    num_images, H, W, num_channels = X.shape
    pos_dir, neg_dir = ensure_dirs()

    # Initialize indices iterator in session state
    if 'indices' not in st.session_state:
        st.session_state.indices = list(random_index_iterator(num_images, num_channels))
        st.session_state.idx = 0


    # Controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Positive"):
            label = 'pos'
        else:
            label = None
    with col2:
        if st.button("Negative"):
            label = 'neg'
    with col3:
        if st.button("Skip"):
            label = 'skip'
    with col4:
        if st.button("Exit"):
            st.stop()

    # Process label action
    if 'label' in locals() and label:
        idx_tuple = st.session_state.indices[st.session_state.idx]
        name = get_image_name(idx_tuple)
        raw = get_image(X, idx_tuple)
        if label == 'pos':
            save_image(raw, name, pos_dir)
        elif label == 'neg':
            save_image(raw, name, neg_dir)
        # advance
        st.session_state.idx += 1

    # Skip already labeled or show next image
    while st.session_state.idx < len(st.session_state.indices):
        idx_tuple = st.session_state.indices[st.session_state.idx]
        name = get_image_name(idx_tuple)
        if is_already_labeled(name, pos_dir, neg_dir):
            st.session_state.idx += 1
            continue
        # display image
        raw = get_image(X, idx_tuple)
        proc = preprocess_image(raw)
        disp = (proc * 255).astype(np.uint8)
        pil = Image.fromarray(disp).resize((600, 600))
        st.image(pil, caption=name)
        break
    else:
        st.success("All images labeled or skipped.")

if __name__ == "__main__":
    main()