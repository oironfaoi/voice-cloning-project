"""Các hàm tiện ích dùng chung cho Voice Cloning Project."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """
    Trả về thư mục gốc của project:
    .../voice-cloning-project
    """
    return Path(__file__).resolve().parent.parent


def resolve_path(path_str: str | None, base_dir: Path | None = None) -> Path | None:
    """
    Chuyển path tương đối -> tuyệt đối theo project root.
    Nếu path đã là tuyệt đối thì giữ nguyên.
    """
    if path_str is None:
        return None

    path = Path(path_str)
    if path.is_absolute():
        return path

    if base_dir is None:
        base_dir = get_project_root()

    return (base_dir / path).resolve()


def load_config(config_path: str | None = None) -> dict:
    """
    Load file YAML config.
    Nếu không truyền config_path thì mặc định dùng:
    <project_root>/configs/settings.yaml
    """
    project_root = get_project_root()

    if config_path is None:
        config_file = project_root / "configs" / "settings.yaml"
    else:
        config_file = resolve_path(config_path, project_root)

    if config_file is None or not config_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Chuẩn hóa các đường dẫn trong config thành tuyệt đối
    paths_cfg = config.get("paths", {})
    for key, value in list(paths_cfg.items()):
        if isinstance(value, str):
            paths_cfg[key] = str(resolve_path(value, project_root))

    # Gắn thêm project_root để dùng ở nơi khác nếu cần
    config["project_root"] = str(project_root)
    return config


def setup_logger(log_dir: str = "outputs/logs") -> logging.Logger:
    """
    Tạo logger ghi ra file + console.
    Tránh nhân đôi handler khi chạy lại nhiều lần trong PyCharm.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"clone_{timestamp}.log"

    logger = logging.getLogger("voice_cloning")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Xóa handler cũ để tránh log bị lặp khi Run nhiều lần trong IDE
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"📝 Log file: {log_file}")
    return logger


def generate_output_filename(
    output_dir: str,
    prefix: str = "clone",
    ext: str = "wav",
) -> str:
    """
    Sinh tên file output theo timestamp.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(output_dir_path / f"{prefix}_{timestamp}.{ext}")