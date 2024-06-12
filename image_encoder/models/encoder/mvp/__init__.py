"""Model zoo (based on Detectron, pycls, and CLIP)."""

import re
import sys
from urllib import request as urlrequest
from pathlib import Path

from .vit import vit_s16, vit_b16, vit_l16


# Model download cache directory
_DOWNLOAD_CACHE = Path("~/.model_checkpoints/mvp").expanduser().resolve()

# Pretrained models
_MODELS = {
    "vits-mae-hoi": "https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth",
    "vits-mae-in": "https://berkeley.box.com/shared/static/qlsjkv03nngu37eyvtjikfe7rz14k66d.pth",
    "vits-sup-in": "https://berkeley.box.com/shared/static/95a4ncqrh1o7llne2b1gpsfip4dt65m4.pth",
    "vitb-mae-egosoup": "https://berkeley.box.com/shared/static/0ckepd2ja3pi570z89ogd899cn387yut.pth",
    "vitl-256-mae-egosoup": "https://berkeley.box.com/shared/static/6p0pc47mlpp4hhwlin2hf035lxlgddxr.pth",
}
_MODEL_FUNCS = {
    "vits": vit_s16,
    "vitb": vit_b16,
    "vitl": vit_l16,
}


def _progress_bar(count, total):
    """Report download progress. Credit:
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


def download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    """Download url and write it to dst_file_path. Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    url = url + "?dl=1" if "dropbox" in url else url
    req = urlrequest.Request(url)
    response = urlrequest.urlopen(req)
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


def maybe_download(url, fpath):
    fpath = Path(fpath)
    if not fpath.exists():
        fpath.parent.mkdir(parents=True, exist_ok=True)
        lock_acquired = False
        lock_file = fpath.with_suffix(".lock")
        if lock_file.exists():
            while lock_file.exists():
                pass
        else:
            lock_file.touch()
            lock_acquired = True
            download_url(url, fpath)
            if lock_acquired:
                lock_file.unlink()


def cache_url(model_name, url_or_file, cache_dir=_DOWNLOAD_CACHE):
    """Download the file specified by the URL to the cache_dir and return the path to
    the cached file. If the argument is not a URL, simply return it as is.
    """
    is_url = re.match(r"^(?:http)s?://", url_or_file, re.IGNORECASE) is not None
    if not is_url:
        return url_or_file
    url = url_or_file
    fname = model_name + ".pth"
    cache_fpath = cache_dir / fname
    maybe_download(url, cache_fpath)
    return cache_fpath.as_posix()


def available_models():
    """Retrieves the names of available models."""
    return list(_MODELS.keys())


def load_mvp(model_name, output_dim=None):
    """Loads a pre-trained model."""
    assert model_name in _MODELS.keys(), "Model {} not available".format(model_name)
    pretrained = cache_url(model_name, _MODELS[model_name])
    model_func = _MODEL_FUNCS[model_name.split("-")[0]]
    img_size = 256 if "-256-" in model_name else 224
    model, _ = model_func(pretrained=pretrained, img_size=img_size)
    return model
