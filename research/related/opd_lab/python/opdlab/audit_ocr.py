"""Audit-only OCR; never imported by training or target selection."""
import os
from pathlib import Path
import subprocess

from .io import read_json, sha256


class AuditOCR:
    def __init__(self, config_path):
        self.config = read_json(config_path)
        for path, expected in self.config["assets_sha256"].items():
            if sha256(path) != expected:
                raise ValueError("Audit OCR executable/model/library changed")
        self.env = dict(os.environ, LD_LIBRARY_PATH=self.config["library_dir"], OMP_THREAD_LIMIT="1")
        self.version = subprocess.check_output([self.config["binary"], "--version"],
                                              env=self.env, stderr=subprocess.STDOUT, text=True).splitlines()[0]

    def __call__(self, image, expected_sha256=None):
        image = Path(image).resolve()
        if expected_sha256 and sha256(image) != expected_sha256:
            raise ValueError("Audit image changed after scoring")
        c = self.config
        command = [c["binary"], str(image), "stdout", "--tessdata-dir", c["tessdata"],
                   "-l", c["language"], "--oem", str(c["oem"]), "--psm", str(c["psm"])]
        result = subprocess.run(command, env=self.env, capture_output=True, text=True, check=True, timeout=30)
        return {"text": result.stdout.strip(), "stderr": result.stderr.strip(), "image_sha256": sha256(image)}
