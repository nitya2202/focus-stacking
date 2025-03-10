# Focus Stacking in Python

This repository contains a Python implementation of focus stacking using multi-scale pyramid blending. Focus stacking is a digital image processing technique that combines multiple images taken at different focus distances to create a single image with a greater depth of field (DoF) than any of the individual source images.

## Overview

Focus stacking is particularly useful in macro photography, microscopy, and landscape photography where the depth of field might be limited. This implementation uses gradient-based blending with Laplacian pyramids to seamlessly combine images with different focus points.

## Features

- Multiple reconstruction methods:
  - Gradient-based blending
  - Laplacian filter-based blending
  - Explicit mask with variable-size median filtering
- Multi-scale pyramid processing for high-quality results
- Support for color and grayscale images
- Channel-specific processing for better control

## Requirements

- Python 3.6+
- NumPy
- OpenCV (cv2)
- scikit-image

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

Run the focus stacking script with input images:

```bash
# Using specific image paths
python focus_stacking.py --back "path/to/background.jpg" --front "path/to/foreground.jpg" --output "path/to/output_folder"

# Example with eye images
python focus_stacking.py --back "examples/sclera.jpg" --front "examples/cornea.jpg" --output "results"
```

3. The script will generate multiple output images showing different blending methods:
   - `focus_stacked_by_channel.jpg` - Result from processing each color channel separately
   - `focus_stacked_all_channels.jpg` - Result from processing all channels together
   - `focus_stacked_new_method.jpg` - Result using variable-size median filtering
   - `blending_mask.jpg` - The blending mask used
   - `focus_stacked_direct.jpg` - Result from direct reconstruction without pyramids

## How It Works

The algorithm follows these steps:

1. Calculate gradient magnitudes for both input images
2. Compare gradients to determine which image has more detail at each pixel
3. Create a blending mask based on gradient differences
4. Smooth the mask to avoid artifacts
5. Blend the images using the mask
6. Improve results using multi-scale (pyramid) processing

## Customization

You can adjust several parameters in the code:

- `beta` in the reconstruction functions controls the sigmoid steepness
- The kernel size in the reconstruction functions controls the smoothing amount
- The number of pyramid levels (default is 5)
- The median filter sizes in `reconstr_new` function

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on a MATLAB implementation of focus stacking using multi-scale pyramid blending.