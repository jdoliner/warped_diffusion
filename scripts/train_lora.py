#!/usr/bin/env python3
"""Script to train LoRA adapters for warped diffusion."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from warped_diffusion.training.train_lora import main

if __name__ == "__main__":
    main()
