#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import sys
from . import vit
import urllib
from pathlib import Path


_EAI_VC1_BASE_URL = "https://dl.fbaipublicfiles.com/eai-vc/"


# progress_bar and download_url from
# https://github.com/facebookresearch/Detectron/blob/1809dd41c1ffc881c0d6b1c16ea38d08894f8b6d/detectron/utils/io.py
def _progress_bar(count, total):
    """Report download progress.
    Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write(
        "  [{}] {}% of {:.1f}MB file  \r".format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write("\n")


def _download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    """Download url and write it to dst_file_path.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(f"Error downloading model from {_EAI_VC1_BASE_URL}:\n{e}")
        raise
    total_size = response.info().get("Content-Length").strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(dst_file_path, "wb") as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)

    return bytes_so_far


def download_model_if_needed(ckpt_file: Path):
    if not ckpt_file.exists():
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        lock_acquired = False
        lock_file = ckpt_file.with_suffix(".lock")
        if lock_file.exists():
            while lock_file.exists():
                pass
        else:
            lock_file.touch()
            lock_acquired = True
            model_name = ckpt_file.name
            model_url = _EAI_VC1_BASE_URL + model_name
            _download_url(model_url, ckpt_file)
            if lock_acquired:
                lock_file.unlink()


def load_model(model_name, output_dim=None):  # output_dim for config interpolation
    if model_name == "vc1_vitb":
        target = vit.vit_base_patch16
    elif model_name == "vc1_vitl":
        target = vit.vit_large_patch16
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model = target(
        img_size=224,
        use_cls=True,
        drop_path_rate=0.0,
    )
    checkpoint_dir = Path("~/.model_checkpoints/vc1").expanduser().resolve()
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"{model_name}.pth"
    download_model_if_needed(checkpoint_path)
    model = vit.load_mae_encoder(model, checkpoint_path)
    return model
