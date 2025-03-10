# Focus Stacking Results Comparison

This document provides a visual comparison of the focus stacking results using different methods implemented in this repository.

## Input Images

The algorithm takes two input images with different focus points:

1. **Background Focus Image** (`sclera.jpg`):
   - This image has the sclera (white part of the eye) in focus
   - The iris and cornea may be slightly out of focus

2. **Foreground Focus Image** (`cornea.jpg`):
   - This image has the cornea and iris in focus
   - The sclera may be slightly out of focus

## Output Comparisons

### All Channels Method
`focus_stacked_all_channels.jpg`
- Processes the entire color image through the pyramid
- Good overall balance
- May have slight color shifts in transition areas

### New Method with Variable Median Filters
`focus_stacked_new_method.jpg`
- Uses different filter sizes at each pyramid level
- Better at handling noise and transitions
- Preserves boundaries between regions more clearly

### Direct Method (No Pyramids)
`focus_stacked_direct.jpg`
- Simple direct blending without multi-scale processing
- Fastest method but may have more visible seams
- Useful as a quick preview

### Blending Mask
`blending_mask.jpg`
- Shows which parts of each input image are used
- Bright areas come from the background image
- Dark areas come from the foreground image
- Useful for debugging and understanding the algorithm

## When to Use Each Method

- **Standard All-Channels Method**: Best for general photography with moderate depth differences
- **New Method with Variable Filters**: Best for scientific/technical images where precision matters
- **Direct Method**: Best for quick previews or when input images have very different exposures

## Tips for Best Results

1. Use a tripod to ensure images are perfectly aligned
2. Use manual focus and manual exposure settings
3. Start with the standard all-channels method first
4. If you see artifacts, try the new method with variable filters
5. Adjust the beta parameter to control transition hardness