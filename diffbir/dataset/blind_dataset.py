from typing import List, Dict, Mapping, Any, Optional
import os
import time
import io
import random

import numpy as np
from PIL import Image
import torch.utils.data as data

from ..dataset.utils import center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class BlindPairedDataset(data.Dataset):
    """Paired dataset for blind pixel (fixed dead pixels) experiments.

    Expects a folder structure under `dataset_path`:
      - train_blur, train_sharp, train_mask (optional)
      - val_blur, val_sharp, val_mask (optional)
      - test_blur, test_sharp, test_mask (optional)

    This dataset returns (gt, lq, prompt) to be compatible with existing
    training loops. For grayscale inputs it will duplicate the single channel
    into 3 channels so models expecting RGB can be used unchanged.
    """

    def __init__(
        self,
        dataset_path: str,
        mode: str = "train",
        file_backend_cfg: Mapping[str, Any] = None,
        out_size: Optional[int] = None,
        crop_type: str = "none",
    ) -> "BlindPairedDataset":
        super(BlindPairedDataset, self).__init__()
        assert mode in ["train", "val", "test"], "mode must be train/val/test"
        self.dataset_path = dataset_path
        self.mode = mode
        self.out_size = out_size
        self.crop_type = crop_type
        # image folders
        suf = {
            "train": "train",
            "val": "val",
            "test": "test",
        }[mode]
        self.blur_dir = os.path.join(dataset_path, f"{suf}_blur")
        self.sharp_dir = os.path.join(dataset_path, f"{suf}_sharp")
        self.mask_dir = os.path.join(dataset_path, f"{suf}_mask")

        # build pairing by filename intersection
        def list_imgs(d):
            if not os.path.isdir(d):
                return []
            exts = {".png", ".jpg", ".jpeg", ".bmp"}
            files = [f for f in sorted(os.listdir(d)) if os.path.splitext(f)[1].lower() in exts]
            return files

        blur_files = list_imgs(self.blur_dir)
        sharp_files = list_imgs(self.sharp_dir)
        # take intersection by basename
        blur_set = {os.path.splitext(f)[0]: f for f in blur_files}
        sharp_set = {os.path.splitext(f)[0]: f for f in sharp_files}
        keys = sorted(list(set(blur_set.keys()) & set(sharp_set.keys())))
        self.pairs = [(blur_set[k], sharp_set[k]) for k in keys]

    def _load_image(self, path: str):
        # path is local filesystem absolute path
        try:
            img = Image.open(path)
        except Exception:
            return None
        # convert to grayscale for robustness then duplicate to 3 channels
        img = img.convert("L")
        arr = np.array(img)
        # cropping/resizing if requested
        if self.out_size is not None and self.crop_type != "none":
            pil_img = Image.fromarray(arr)
            if pil_img.height == self.out_size and pil_img.width == self.out_size:
                arr = np.array(pil_img)
            else:
                if self.crop_type == "center":
                    arr = center_crop_arr(pil_img, self.out_size)
                elif self.crop_type == "random":
                    arr = random_crop_arr(pil_img, self.out_size, min_crop_frac=0.8)

        # duplicate to 3 channels (H,W,3)
        arr3 = np.stack([arr, arr, arr], axis=2).astype(np.uint8)
        return arr3

    def __getitem__(self, index: int):
        blur_name, sharp_name = self.pairs[index]
        blur_path = os.path.join(self.blur_dir, blur_name)
        sharp_path = os.path.join(self.sharp_dir, sharp_name)

        img_lq = self._load_image(blur_path)
        img_gt = self._load_image(sharp_path)
        if img_lq is None or img_gt is None:
            # fallback to a random sample
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)

        # follow CodeformerDataset conventions: img arrays are HWC, PIL/RGB
        # convert to float formats expected by training code
        # BGR ordering used in code elsewhere; keep same transformations
        gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        lq = img_lq[..., ::-1].astype(np.float32)

        # gt [-1, 1], lq [0,1]
        gt = (gt * 2 - 1).astype(np.float32)

        prompt = ""
        return gt, lq, prompt

    def __len__(self):
        return len(self.pairs)
