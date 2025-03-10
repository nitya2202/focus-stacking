# Focus Stacking Documentation

## Theory

Focus stacking is a digital image processing technique that combines multiple images taken at different focus distances to extend the depth of field in the final image. This is particularly useful in:

- **Macro photography**: Where depth of field is extremely shallow
- **Microscopy**: Where different planes of a sample need to be in focus
- **Landscape photography**: When both near and far elements need to be sharp

## Algorithm Overview

This implementation uses gradient-based blending with Laplacian pyramids:

1. **Gradient Calculation**: Compute gradient magnitudes in both images to identify which regions are in better focus
2. **Mask Creation**: Generate a blending mask based on which image has stronger gradients at each pixel
3. **Multi-scale Processing**: Use Laplacian pyramids to blend at multiple scales, preserving both fine and coarse details
4. **Final Blending**: Combine the results from different scales to produce the final image

## Implementation Details

### Reconstruction Methods

The code implements three different reconstruction methods:

1. **Standard Gradient-based Blending** (`reconstr`):
   - Uses horizontal and vertical gradient filters
   - Applies a sigmoid function to create a soft blending mask
   - Best for general purpose blending

2. **Laplacian Filter-based Blending** (`reconstr_old`):
   - Uses a Laplacian filter to detect edges
   - Adds Gaussian smoothing to the mask
   - Good for highlighting fine details

3. **Explicit Mask with Median Filtering** (`reconstr_new`):
   - Creates a binary mask based on gradient comparison
   - Uses variable-size median filtering to clean up the mask
   - Adaptive filter sizes for different pyramid levels
   - Best for handling noise and preserving boundaries

### Laplacian Pyramids

The algorithm uses Laplacian pyramids to:
- Process the image at multiple scales
- Capture both fine details and large structures
- Avoid artifacts from direct blending

## Eye Image Example

The repository includes example images of an eye:
- `sclera.jpg`: Image focused on the sclera (white part of the eye)
- `cornea.jpg`: Image focused on the cornea and iris

The focus stacking process combines these to create an all-in-focus image where both the sclera and the cornea/iris are sharp.

## Parameter Tuning

Several parameters can be adjusted for different results:

- `beta`: Controls the steepness of the sigmoid function (lower values create softer transitions)
- Filter sizes: Larger filters create smoother transitions but may lose some detail
- Pyramid levels: More levels can capture more scales but increase processing time

## Troubleshooting

Common issues:
- Halos around high-contrast edges
- Ghosting when input images are not perfectly aligned
- Artifacts from moving elements between shots

Solutions:
- Adjust the beta parameter for different transition hardness
- Use image alignment preprocessing if inputs are not aligned
- Try different reconstruction methods for different types of images