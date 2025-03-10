import numpy as np
import cv2
import os
import argparse

def reconstr(img_b, img_f):
    """
    Main reconstruction function using gradient-based blending
    
    Parameters:
    img_b (ndarray): Background image
    img_f (ndarray): Foreground image
    
    Returns:
    ndarray: Blended image
    """
    # Define edge detection filters for horizontal and vertical gradients
    h1 = np.array([-1, 2, -1])  # Horizontal gradient filter
    h2 = np.array([-1, 2, -1]).reshape(3, 1)  # Vertical gradient filter
    
    # Calculate gradient magnitudes for both images
    Gb_h = cv2.filter2D(img_b, -1, h1)
    Gb_v = cv2.filter2D(img_b, -1, h2)
    Gb = np.abs(Gb_h) + np.abs(Gb_v)  # Background gradients
    
    Gf_h = cv2.filter2D(img_f, -1, h1)
    Gf_v = cv2.filter2D(img_f, -1, h2)
    Gf = np.abs(Gf_h) + np.abs(Gf_v)  # Foreground gradients
    
    # Calculate difference map between gradients
    if len(img_b.shape) > 2:  # Color image
        M = np.sum(Gb - Gf, axis=2)  # Sum across color channels to get 2D array
    else:  # Grayscale image
        M = Gb - Gf
    
    # Apply majority filter to smooth the mask
    k = 3  # kernel size for smoothing
    kernel = np.ones((k, k))
    M_filter = cv2.filter2D(M, -1, kernel)
    
    # Apply sigmoid function to create soft mask (with protection against overflow)
    beta = 0.1  # Controls steepness of sigmoid (reduced to prevent overflow)
    M_filter_clipped = np.clip(M_filter, -50, 50)  # Clip to avoid exp overflow
    M_hat = 1 / (1 + np.exp(-beta * M_filter_clipped))
    
    # Reshape mask for color images
    if len(img_b.shape) > 2:
        M_hat = np.repeat(M_hat[:, :, np.newaxis], 3, axis=2)
    
    # Blend images using the mask
    L = M_hat * img_b + (1 - M_hat) * img_f
    
    return L

def reconstr_old(img_b, img_f):
    """
    Older version using Laplacian filter
    
    Parameters:
    img_b (ndarray): Background image
    img_f (ndarray): Foreground image
    
    Returns:
    ndarray: Blended image
    """
    # Use Laplacian filter
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    Gb = cv2.filter2D(img_b, -1, laplacian)
    Gf = cv2.filter2D(img_f, -1, laplacian)
    
    if len(img_b.shape) > 2:  # Color image
        M = np.sum(Gb - Gf, axis=2)
    else:  # Grayscale image
        M = Gb - Gf
    
    # Apply majority filter
    k = 3
    kernel = np.ones((k, k))
    M_filter = cv2.filter2D(M, -1, kernel)
    
    # Create soft mask using sigmoid (with protection against overflow)
    beta = 0.1  # Reduced to prevent overflow
    M_filter_clipped = np.clip(M_filter, -50, 50)  # Clip to avoid exp overflow
    M_hat = 1 / (1 + np.exp(-beta * M_filter_clipped))
    
    # Additional Gaussian smoothing on mask
    M_hat_s = cv2.GaussianBlur(M_hat, (0, 0), 3)
    
    # Reshape mask for color images
    if len(img_b.shape) > 2:
        M_hat_s = np.repeat(M_hat_s[:, :, np.newaxis], 3, axis=2)
    
    # Final blend
    L = M_hat_s * img_b + (1 - M_hat_s) * img_f
    
    return L

def reconstr_new(img_b, img_f, n):
    """
    New version with explicit mask generation and variable-size median filtering
    
    Parameters:
    img_b (ndarray): Background image
    img_f (ndarray): Foreground image
    n (int): Size of median filter (must be odd)
    
    Returns:
    ndarray: Blended image
    ndarray: Blending mask
    """
    # Ensure n is odd (required by medianBlur)
    n = n if n % 2 == 1 else n + 1
    
    # Calculate gradients
    h1 = np.array([-1, 2, -1])
    h2 = np.array([-1, 2, -1]).reshape(3, 1)
    
    Gb_h = cv2.filter2D(img_b, -1, h1)
    Gb_v = cv2.filter2D(img_b, -1, h2)
    Gb = np.abs(Gb_h) + np.abs(Gb_v)
    
    Gf_h = cv2.filter2D(img_f, -1, h1)
    Gf_v = cv2.filter2D(img_f, -1, h2)
    Gf = np.abs(Gf_h) + np.abs(Gf_v)
    
    # Convert to 2D arrays
    if len(img_b.shape) > 2:  # Color image
        Gb_2d = np.sum(Gb, axis=2)
        Gf_2d = np.sum(Gf, axis=2)
    else:
        Gb_2d = Gb
        Gf_2d = Gf
    
    # Apply majority filter to gradients
    k = 3
    kernel = np.ones((k, k))
    Gb_filter = cv2.filter2D(Gb_2d, -1, kernel)
    Gf_filter = cv2.filter2D(Gf_2d, -1, kernel)
    
    # Create binary mask based on gradient comparison
    mask = np.zeros(Gb_2d.shape, dtype=np.uint8)
    mask[np.abs(Gb_filter) > np.abs(Gf_filter)] = 1
    
    # Clean up mask using median filter
    # OpenCV's medianBlur for this case requires uint8 single-channel images
    # Convert to uint8 as required by OpenCV's medianBlur
    mask_uint8 = mask.astype(np.uint8)
    mask_blurred = cv2.medianBlur(mask_uint8, n)
    # Convert back to float for blending
    mask_blurred = mask_blurred.astype(np.float32) / 255.0
    
    # Extend mask to match image dimensions if color image
    if len(img_b.shape) > 2:
        mask_3d = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)
    else:
        mask_3d = mask_blurred
    
    # Final blend
    L = mask_3d * img_b + (1 - mask_3d) * img_f
    
    return L, mask_3d

def generate_laplacian_pyramid(img, levels):
    """
    Generate a Laplacian pyramid for an image
    
    Parameters:
    img (ndarray): Input image
    levels (int): Number of pyramid levels
    
    Returns:
    list: Laplacian pyramid levels
    """
    pyramid = []
    current = img.copy()
    
    for i in range(levels-1):
        # Generate next level by downsampling
        down = cv2.pyrDown(current)
        # Expand back up to original size
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        # Laplacian is the difference between current level and expanded next level
        pyramid.append(current - up)
        # Move to next level
        current = down
    
    # Add smallest level to pyramid
    pyramid.append(current)
    
    return pyramid

def pyr_expand(img):
    """
    Expand an image to double its size
    
    Parameters:
    img (ndarray): Input image
    
    Returns:
    ndarray: Expanded image
    """
    return cv2.pyrUp(img)

def main(back_path=None, front_path=None, output_dir=None):
    """
    Main function to demonstrate focus stacking
    
    Parameters:
    back_path (str): Path to background (far focus) image
    front_path (str): Path to foreground (near focus) image
    output_dir (str): Directory to save output images
    """
    # Set default paths if not provided
    if back_path is None:
        back_path = "back.jpg"
    if front_path is None:
        front_path = "front.jpg"
    if output_dir is None:
        output_dir = "."
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input images
    try:
        print(f"Loading images from:\n  - {back_path}\n  - {front_path}")
        back = cv2.imread(back_path)
        front = cv2.imread(front_path)
        
        if back is None:
            print(f"Error: Could not load background image from {back_path}")
            return
        if front is None:
            print(f"Error: Could not load foreground image from {front_path}")
            return
            
        # Check if images have the same dimensions
        if back.shape != front.shape:
            print(f"Warning: Images have different dimensions. Background: {back.shape}, Foreground: {front.shape}")
            print("Resizing foreground image to match background...")
            front = cv2.resize(front, (back.shape[1], back.shape[0]))
            
        back = back.astype(np.float64)
        front = front.astype(np.float64)
        print("Images loaded successfully")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Alternative approach: Process all channels together
    # Generate Laplacian pyramids for full color images
    print("Generating Laplacian pyramids...")
    back_l = generate_laplacian_pyramid(back, 5)
    front_l = generate_laplacian_pyramid(front, 5)
    
    print("Processing pyramid levels...")
    # Process full color image through pyramid levels
    C4 = reconstr(back_l[4], front_l[4])
    C3 = reconstr(back_l[3], front_l[3]) + pyr_expand(C4)
    C2 = reconstr(back_l[2], front_l[2]) + pyr_expand(C3)
    C1 = reconstr(back_l[1], front_l[1]) + pyr_expand(C2)
    C0 = reconstr(back_l[0], front_l[0]) + pyr_expand(C1)
    
    # Save the result
    C0_8bit = np.clip(C0, 0, 255).astype(np.uint8)
    output_path = os.path.join(output_dir, "focus_stacked_all_channels.jpg")
    cv2.imwrite(output_path, C0_8bit)
    print(f"Saved result to: {output_path}")
    
    # Try new reconstruction method with different median filter sizes for each level
    print("Trying alternative reconstruction method...")
    # Using odd-sized filters as required by OpenCV's medianBlur
    C4, M4 = reconstr_new(back_l[4], front_l[4], 3)     # Small filter for smallest level
    C3, M3 = reconstr_new(back_l[3], front_l[3], 5)     # Increasing filter sizes
    C2, M2 = reconstr_new(back_l[2], front_l[2], 9)     # for larger levels
    C1, M1 = reconstr_new(back_l[1], front_l[1], 19)
    C0, M0 = reconstr_new(back_l[0], front_l[0], 35)    # Largest filter for base level
    
    # Reconstruct final image from new method results
    L4 = C4
    L3 = C3 + pyr_expand(L4)
    L2 = C2 + pyr_expand(L3)
    L1 = C1 + pyr_expand(L2)
    L0 = C0 + pyr_expand(L1)
    
    # Save result from new method
    L0_8bit = np.clip(L0, 0, 255).astype(np.uint8)
    output_path = os.path.join(output_dir, "focus_stacked_new_method.jpg")
    cv2.imwrite(output_path, L0_8bit)
    print(f"Saved result to: {output_path}")
    
    # Save blending mask
    if len(M0.shape) > 2:
        M0_gray = cv2.cvtColor(np.clip(M0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        M0_gray = np.clip(M0, 0, 255).astype(np.uint8)
    output_path = os.path.join(output_dir, "blending_mask.jpg")
    cv2.imwrite(output_path, M0_gray)
    print(f"Saved mask to: {output_path}")
    
    # Alternative: Direct reconstruction without pyramids
    print("Creating direct reconstruction without pyramids...")
    C_direct = reconstr(back, front)
    C_direct_8bit = np.clip(C_direct, 0, 255).astype(np.uint8)
    output_path = os.path.join(output_dir, "focus_stacked_direct.jpg")
    cv2.imwrite(output_path, C_direct_8bit)
    print(f"Saved direct result to: {output_path}")
    
    print("Focus stacking completed successfully!")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Focus stacking of two images.')
    parser.add_argument('--back', type=str, help='Path to background (far focus) image')
    parser.add_argument('--front', type=str, help='Path to foreground (near focus) image')
    parser.add_argument('--output', type=str, help='Directory to save output images')
    args = parser.parse_args()
    
    # Run main function with provided arguments
    main(back_path=args.back, front_path=args.front, output_dir=args.output)