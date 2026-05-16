#!/usr/bin/env python3
"""Download the Hateful Memes dataset with KaggleHub."""

from __future__ import annotations

import kagglehub


path = kagglehub.dataset_download("parthplc/facebook-hateful-meme-dataset")
print("Path to dataset files:", path)
