import os
import cv2
import numpy as np
from PIL import Image

def apply_watermark(input_folder, output_folder, watermark_path, padding=10, transparency=1.0, scale_factor=0.2):
    """
    Applies a watermark to all images in a folder with fixed scaling.
    
    Parameters:
    - input_folder: Path to the folder containing images.
    - output_folder: Path to the folder where watermarked images will be saved.
    - watermark_path: Path to the watermark PNG file.
    - padding: Space between the watermark and image edges.
    - transparency: Watermark transparency (0.0 to 1.0).
    - max_scale_factor: Max scale factor for watermark size (between 0.0 and 1.0).
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load watermark with alpha channel
    watermark = Image.open(watermark_path).convert("RGBA")
    wm_width, wm_height = watermark.size
    
    # Apply scaling to the watermark
    wm_width = int(wm_width * scale_factor)
    wm_height = int(wm_height * scale_factor)
    watermark_resized = watermark.resize((wm_width, wm_height), Image.Resampling.LANCZOS)
    
    # Apply transparency
    alpha = watermark_resized.split()[3].point(lambda p: int(p * transparency))
    watermark_resized.putalpha(alpha)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Load image
            image = Image.open(image_path).convert("RGBA")
            img_width, img_height = image.size

            
            # Position watermark at center
            x = (img_width - wm_width) // 2
            y = (img_height - wm_height) // 2
            
            # Apply watermark
            watermarked = Image.new("RGBA", image.size, (0, 0, 0, 0))
            watermarked.paste(image, (0, 0))
            watermarked.paste(watermark_resized, (x, y), watermark_resized)
            
            # Convert back to RGB and save
            final_image = watermarked.convert("RGB")
            final_image.save(output_path)
            print(f"Watermark applied: {output_path}")

if __name__ == "__main__":
    input_folder = os.path.join(os.getcwd(), "images\\input")  # Input folder
    output_folder = os.path.join(os.getcwd(), "images\\output\\watermarked")  # Output folder
    watermark_path = os.path.join(os.getcwd(), "watermark.png") # Watermark path
    
    apply_watermark(input_folder, output_folder, watermark_path, padding=10, transparency=0.075, scale_factor=0.1)
