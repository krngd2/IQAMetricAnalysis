import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def apply_gaussian_blur(image_path, kernel_size=(5, 5), sigma_x=0):
    """Applies Gaussian blur to an image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        blurred_img = cv2.GaussianBlur(img, kernel_size, sigma_x)
        # Save to a temporary in-memory buffer to simulate saving and reloading
        # This ensures the image format is consistent (e.g., for JPEG compression later)
        is_success, buffer = cv2.imencode(".png", blurred_img) # Using PNG to avoid lossy compression here
        if not is_success:
            raise ValueError("Failed to encode blurred image.")
        # Convert buffer to PIL Image
        pil_image = Image.open(BytesIO(buffer)).convert('RGB')
        return pil_image
    except Exception as e:
        print(f"Error in apply_gaussian_blur for {image_path}: {e}")
        return None

def apply_gaussian_noise(image_path, mean=0, sigma=25):
    """Applies Gaussian noise to an image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(mean, sigma, img_array.shape).astype(np.float32)
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        is_success, buffer = cv2.imencode(".png", noisy_img_array)
        if not is_success:
            raise ValueError("Failed to encode noisy image.")
        pil_image = Image.open(BytesIO(buffer)).convert('RGB')
        return pil_image
    except Exception as e:
        print(f"Error in apply_gaussian_noise for {image_path}: {e}")
        return None

def apply_jpeg_compression(image_path, quality=75):
    """Applies JPEG compression to an image."""
    try:
        img = Image.open(image_path).convert('RGB') # Ensure image is in RGB
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer).convert('RGB') # Re-open to ensure it's a PIL image
        return compressed_img
    except Exception as e:
        print(f"Error in apply_jpeg_compression for {image_path}: {e}")
        return None
