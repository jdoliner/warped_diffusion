#!/usr/bin/env python3
"""Script to train the Latent-CLIP projection head."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from warped_diffusion.training.train_latent_clip import main

if __name__ == "__main__":
    main()
