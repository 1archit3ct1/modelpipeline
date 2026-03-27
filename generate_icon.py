#!/usr/bin/env python3
"""
Generate a simple ICO icon for Tauri Windows build.
Creates a 256x256 blue gradient square icon.
"""

from PIL import Image, ImageDraw
import os

def create_icon(output_path: str, size: int = 256):
    """Create a simple gradient icon."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a rounded rectangle with gradient-like effect
    margin = size // 8
    for i in range(size // 2 - margin):
        r = int(41 + i * 0.3)  # Blue gradient
        g = int(127 + i * 0.2)  # Green gradient  
        b = int(243 - i * 0.1)  # Blue fade
        draw.rounded_rectangle(
            [margin + i, margin + i, size - margin - i, size - margin - i],
            radius=size // 16,
            fill=(r, g, b, 255)
        )
    
    # Add a simple "A" letter in white
    font_size = size // 2
    # Draw simple geometric "A" shape
    cx, cy = size // 2, size // 2
    leg_width = max(2, size // 20)
    leg_length = size // 3
    
    # Left leg
    draw.line(
        [(cx - leg_length // 2, cy + leg_length // 2), 
         (cx, cy - leg_length // 2)],
        fill=(255, 255, 255, 255),
        width=leg_width
    )
    
    # Right leg
    draw.line(
        [(cx + leg_length // 2, cy + leg_length // 2), 
         (cx, cy - leg_length // 2)],
        fill=(255, 255, 255, 255),
        width=leg_width
    )
    
    # Cross bar
    bar_y = cy
    bar_width = leg_length // 2
    draw.line(
        [(cx - bar_width // 2, bar_y), 
         (cx + bar_width // 2, bar_y)],
        fill=(255, 255, 255, 255),
        width=leg_width
    )
    
    img.save(output_path, format='ICO', sizes=[(256, 256)])
    print(f"Created icon: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "ui", "src-tauri", "icons")
    os.makedirs(icons_dir, exist_ok=True)
    
    icon_path = os.path.join(icons_dir, "icon.ico")
    create_icon(icon_path)
    print(f"Icon saved to: {icon_path}")
