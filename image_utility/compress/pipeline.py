"""
Batch-process product images: home thumbnails (~159×127, ~30KB) and
info views (longest side 1200–2000px, ≤1MB JPEG).
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO

LOGGER_NAME = "image_utility.compress"
ENV_INPUT_DIR = "IMAGE_UTIL_INPUT_DIR"
ENV_LOG_DIR = "IMAGE_UTIL_LOG_DIR"
ENV_MAX_FILES = "IMAGE_UTIL_MAX_FILES"
DEFAULT_LOG_DIR = "log"
LOG_FILENAME = "compress.log"
_COMPRESS_DIR = Path(__file__).resolve().parent

HOME_W, HOME_H = 300, 240
HOME_SIZE = (240, 190)  # slightly larger for better sharpness
HOME_TARGET_BYTES = 30 * 1024
INFO_LONG_MIN, INFO_LONG_MAX = 1200, 2000
INFO_MAX_BYTES = 1024 * 1024

JPEG_EXTS = {".jpg", ".jpeg"}


def _parse_max_files() -> int | None:
    raw = os.getenv(ENV_MAX_FILES, "").strip()
    if not raw:
        return None
    try:
        n = int(raw, 10)
    except ValueError:
        return None
    if n <= 0:
        return None
    return n


def _is_jpeg_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in JPEG_EXTS


def _open_rgb(path: Path) -> Image.Image | None:
    try:
        im = Image.open(path)
        im.load()
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    if im.mode == "P":
        if "transparency" in im.info:
            im = im.convert("RGBA")
        else:
            return im.convert("RGB")
    if im.mode == "RGBA":
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])
        im = background
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _jpeg_bytes(im, quality, subsampling):
    buf = BytesIO()
    im.save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=True,
        progressive=True,
    )
    return buf.getvalue()


def _save_jpeg_near_target(im, out_path: Path, max_bytes=None, *, aggressive=False):
    """
    Ignore size. Save high-quality JPEG.
    """

    data = _jpeg_bytes(im, quality=95, subsampling=0)
    out_path.write_bytes(data)


def _fit_home(im):
    im = im.copy()
    im.thumbnail((600, 600), Image.Resampling.LANCZOS)
    return im


def _resize_info(im: Image.Image) -> Image.Image:
    w, h = im.size
    long_side = max(w, h)
    if long_side == 0:
        return im
    if INFO_LONG_MIN <= long_side <= INFO_LONG_MAX:
        return im
    if long_side > INFO_LONG_MAX:
        scale = INFO_LONG_MAX / long_side
    else:
        scale = INFO_LONG_MIN / long_side
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def setup_logging() -> logging.Logger:
    """Log to console and to ``<log_dir>/compress.log`` (``IMAGE_UTIL_LOG_DIR``)."""
    raw = os.getenv(ENV_LOG_DIR, DEFAULT_LOG_DIR).strip()
    log_dir = Path(raw).expanduser()
    if not log_dir.is_absolute():
        log_dir = Path.cwd() / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILENAME

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info("Logging to %s", log_file)
    return logger


def _output_dirs(input_dir: Path) -> tuple[Path, Path]:
    parent = input_dir.parent
    name = input_dir.name
    home = parent / f"{name}_compressed_home"
    info = parent / f"{name}_compressed_info"
    return home, info


def process_folder(
        input_dir: Path,
        *,
        logger: logging.Logger | None = None,
        max_files: int | None = None,
) -> tuple[int, int]:
    """
    Process all JPEGs in input_dir. Returns (ok_count, skipped_count).
    Stops after ``max_files`` successful compressions when set.
    """
    log = logger or logging.getLogger(LOGGER_NAME)
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    log.info("Input directory: %s", input_dir)
    if max_files is not None:
        log.info("Max files (successful compressions): %s", max_files)
    home_dir, info_dir = _output_dirs(input_dir)
    home_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output home: %s", home_dir)
    log.info("Output info: %s", info_dir)

    files = sorted(p for p in input_dir.iterdir() if _is_jpeg_file(p))
    ok, skipped = 0, 0

    for path in files:
        if max_files is not None and ok >= max_files:
            log.info(
                "Stopped: reached %s=%s (%s file(s) compressed).",
                ENV_MAX_FILES,
                max_files,
                ok,
            )
            break

        im = _open_rgb(path)
        if im is None:
            log.warning("Skip (unreadable or corrupt): %s", path.name)
            skipped += 1
            continue
        try:
            home = _fit_home(im)
            home_out = home_dir / path.name
            _save_jpeg_near_target(
                home, home_out, HOME_TARGET_BYTES, aggressive=True
            )

            info = _resize_info(im)
            info_out = info_dir / path.name
            _save_jpeg_near_target(
                info, info_out, INFO_MAX_BYTES, aggressive=False
            )
            ok += 1
            home_sz = home_out.stat().st_size
            info_sz = info_out.stat().st_size
            log.info(
                "Compressed %s | home=%s (%d B) | info=%s (%d B)",
                path.name,
                home_out.name,
                home_sz,
                info_out.name,
                info_sz,
            )
        except OSError as e:
            log.warning("Skip (write error) %s: %s", path.name, e)
            skipped += 1
        finally:
            im.close()

    return ok, skipped


def main() -> int:
    load_dotenv(_COMPRESS_DIR / ".env")
    logger = setup_logging()

    env_path = os.getenv(ENV_INPUT_DIR, "").strip()
    if not env_path:
        logger.error("%s is not set in .env.", ENV_INPUT_DIR)
        return 1
    folder = Path(env_path)
    max_files = _parse_max_files()
    try:
        ok, skipped = process_folder(folder, logger=logger, max_files=max_files)
    except NotADirectoryError as e:
        logger = logging.getLogger(LOGGER_NAME)
        logger.error("%s", e)
        return 1

    logging.getLogger(LOGGER_NAME).info(
        "Done. %s image(s) written, %s skipped.", ok, skipped
    )
    return 0
