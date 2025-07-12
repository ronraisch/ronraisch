from pathlib import Path
import random
from typing import Optional, Union
import numpy as np
from PIL import Image
import cv2
import streamlit as st

 

ROOT_DIR = Path('/home/ronraisch')
BASE_DIR = Path(f'{ROOT_DIR}/data/labeled')

IMAGE_FILE_TYPE = 'png'


class ImageLabeler:
    def __init__(self, x_path: Path, output_dir: Path, size=600, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.x_path = x_path
        self.output_dir = Path(output_dir)
        self._size = size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

        self.X = self._load_data()
        self.pos_dir, self.neg_dir = self._ensure_dirs()

        self.num_images, self.H, self.W, self.num_channels = self.X.shape
        self.init_indices()

    @property
    def size(self) -> tuple[int, int]:
        if isinstance(self._size, int):
            return (self._size, self._size)
        elif isinstance(self._size, tuple) and len(self._size) == 2:
            return self._size
        raise ValueError("Size must be an int or a tuple of two ints (width, height).")
    
    @size.setter
    def size(self, value: Union[int, tuple[int, int]]):
        if isinstance(value, int) or (isinstance(value, tuple) and len(value) == 2):
            self._size = value
        else:
            raise ValueError("Size must be an int or a tuple of two ints (width, height).")

    def init_indices(self):
        self.indices = random.sample(
            [(i, j) for i in range(self.num_images) for j in range(self.num_channels)],
            k=self.num_images * self.num_channels
        )
        self.idx = 0
        self._prefetch_next()

    def _load_data(self) -> np.ndarray:
        return np.load(self.x_path)

    def _ensure_dirs(self) -> tuple[Path, Path]:
        pos = self.output_dir / "positive"
        neg = self.output_dir / "negative"
        pos.mkdir(parents=True, exist_ok=True)
        neg.mkdir(parents=True, exist_ok=True)
        return pos, neg

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        mi, ma = float(img.min()), float(img.max())
        if ma > mi:
            img = (img - mi) / (ma - mi) * 255.0
        else:
            img = np.zeros_like(img)
        return img.astype(np.uint8)

    def _preprocess(self, raw: np.ndarray) -> Image.Image:
        uint8 = self._normalize(raw)
        if uint8.ndim == 3 and uint8.shape[2] > 1:
            uint8 = cv2.cvtColor(uint8, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(uint8, self.size, interpolation=cv2.INTER_LINEAR)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_img = clahe.apply(resized)
        return Image.fromarray(clahe_img.astype(np.uint8))

    def _get_raw(self, index: tuple) -> np.ndarray:
        i, c = index
        return self.X[i, :, :, c]

    def _get_name(self, index: tuple) -> str:
        i, c = index
        return f"image_{i}_channel_{c}"

    def _is_labeled(self, name: str) -> bool:
        return (self.pos_dir/f"{name}.{IMAGE_FILE_TYPE}").exists() or \
               (self.neg_dir/f"{name}.{IMAGE_FILE_TYPE}").exists()

    def _save(self, raw: np.ndarray, name: str, label: str):
        uint8 = self._normalize(raw)
        target = self.pos_dir if label == 'pos' else self.neg_dir
        Image.fromarray(uint8).save(target / f"{name}.{IMAGE_FILE_TYPE}")

    def _prefetch_next(self):
        # move to next unlabeled
        while self.idx < len(self.indices):
            idx = self.indices[self.idx]
            name = self._get_name(idx)
            if not self._is_labeled(name):
                raw = self._get_raw(idx)
                self.next_raw = raw
                self.next_proc = self._preprocess(raw)
                self.next_name = name
                return
            self.idx += 1
        self.next_raw = None
        self.next_proc = None
        self.next_name = None

    def label_current(self, label: Optional[str]):
        if label in ('pos', 'neg') and self.next_raw is not None and self.next_name is not None:
            self._save(self.next_raw, self.next_name, label)
        self.idx += 1
        self._prefetch_next()

    def has_next(self) -> bool:
        return self.next_proc is not None

    def display(self):
        if self.has_next() and self.next_proc is not None:

            st.image(self.next_proc, caption=self.next_name)
        else:
            st.success("All images labeled or skipped.")


def main():
    st.title("Image Labeling Tool")
    ROOT = Path('/home/ronraisch')
    x_path = ROOT / 'data' / 'processed' / 'X.npy'
    output_dir = ROOT / 'data' / 'labeled'

    if 'labeler' not in st.session_state:
        st.session_state.labeler = ImageLabeler(x_path, output_dir)
    labeler = st.session_state.labeler

    cols = st.columns(5)
    label = None
    with cols[0]:
        if st.button("Positive"): label = 'pos'
    with cols[1]:
        if st.button("Negative"): label = 'neg'
    with cols[2]:
        if st.button("Skip"): label = 'skip'
    with cols[3]:
        if st.button("Exit"): st.stop()
    with cols[4]:
        # add slider to determine size of the image
        labeler.size = int(st.slider("Image Size", 100, 1000, 600))

    if label:
        if label != 'skip':
            labeler.label_current(label)
        else:
            labeler.idx += 1
            labeler._prefetch_next()

    labeler.display()

if __name__ == '__main__':
    main()
