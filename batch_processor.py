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


def document_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(
            (".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt")
        ):
            yield filepath


def zip_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(
            (".zip", ".rar", ".7z", ".tar", ".gz", ".bz2")
        ):
            yield filepath


import mimetypes


def unknown_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and not mimetypes.guess_type(filepath)[0] in [
            i
            for i in [
                "image/png",
                "image/jpeg",
                "image/gif",
                "image/bmp",
                "video/mp4",
                "video/mov",
                "video/avi",
                "video/mkv",
                "video/webm",
                "audio/mp3",
                "audio/wav",
                "audio/flac",
                "audio/ogg",
                "audio/aac",
                "application/pdf",
                "application/docx",
                "application/doc",
                "text/plain",
                "application/rtf",
                "application/odt",
                "application/zip",
                "application/rar",
                "application/x-7z-compressed",
                "application/x-tar",
                "application/gzip",
                "application/bzip2",
            ]
        ]:
            yield filepath
