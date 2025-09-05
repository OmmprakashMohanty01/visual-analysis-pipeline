import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def image_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp")
        ):
            yield filepath


def video_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(
            (".mp4", ".mov", ".avi", ".mkv", ".webm")
        ):
            yield filepath


def audio_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(
            (".mp3", ".wav", ".flac", ".ogg", ".aac")
        ):
            yield filepath
