"""Keep BasicSR 1.4.2's old torchvision import working on newer torchvision."""

import sys

try:
    from torchvision.transforms import functional_tensor  # noqa: F401
except ImportError:
    from torchvision.transforms import functional

    # BasicSR only imports rgb_to_grayscale from this removed module.
    sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)
