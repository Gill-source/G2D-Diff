"""
Lightweight placeholder generator for test/eval flows.

The original repository references `src.generator.G2D_Generator`, but the
implementation is absent. This minimal drop-in provides a `sample` method so
test scripts can run end-to-end without import errors. It currently returns a
set of simple, valid SMILES strings; if you have a trained generator/decoder,
replace the logic in `sample` with the real implementation.
"""

from typing import List
import random


class G2D_Generator:
    @staticmethod
    def sample(diffusion_model=None, condition_vec=None, num_samples: int = 4, device="cpu") -> List[str]:
        """Return a small list of simple SMILES strings as a placeholder."""
        # Small pool of valid, simple molecules
        base_smiles = [
            "C", "CC", "CCC", "c1ccccc1", "C1CC1", "O", "N", "CO", "CN", "CCO",
            "CCN", "C=O", "CC=O", "CCCl", "CCBr"
        ]
        return [random.choice(base_smiles) for _ in range(num_samples)]

