import os
import sys
from OCC.Display.SimpleGui import init_display
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from PIL import Image, ImageDraw, ImageFont


# 1. Load CAD model
if len(sys.argv) != 2:
    print("Usage: python render_occ.py <step_file>")
    sys.exit(1)

step_file = sys.argv[1]
if not os.path.isfile(step_file):
    print(f"Error: File '{step_file}' does not exist.")
    sys.exit(1)

step_file_dir = os.path.dirname(os.path.abspath(step_file))
shape = read_step_file(step_file)

# 2. Initialize viewer
display, start_display, add_menu, add_function_to_menu = init_display('pyqt5')
white = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
display.View.SetBgGradientColors(white, white, 0)
display.View.SetBackgroundColor(white)

# 3. Define views
display.DisplayShape(shape)
display.FitAll()
display.View.TriedronErase()
view_directions = {
    "Isometric": (1, 1, 1),
    "+Y": (0, 1, 0),
    "-Y": (0, -1, 0),
    "+Z": (0, 0, 1),
    "-Z": (0, 0, -1),
    "+X": (1, 0, 0),
    "-X": (-1, 0, 0),
}

# 4. Render views to images
for name, direction in view_directions.items():
    display.View.SetProj(*direction)
    display.View.ZFitAll()
    display.FitAll()
    output_path = os.path.join(step_file_dir, f"{name}.png")
    display.View.Dump(output_path)

display.EraseAll()

# 5. Grid layout
grid = [
    ['Isometric', None, '+Y', None],
    ['-Z', '-X', '+Z', '+X'],
    [None, None, '-Y', None]
]

img_size = (192, 192)
bg_color = (255, 255, 255)
padding = 5
overlay_offset = 30  # extra space at top so text isn't clipped

# 6. Load font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=24)
except:
    font = ImageFont.load_default()

# 7. Resize & pad images
def resize_and_pad(img, size=(256, 256), bg_color=(255, 255, 255)):
    img.thumbnail(size, Image.LANCZOS)
    padded = Image.new("RGB", size, bg_color)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    padded.paste(img, (x, y))
    return padded

# 8. Canvas dimensions
cols = len(grid[0])
rows = len(grid)
cell_width = img_size[0]
cell_height = img_size[1]
combined_width = cols * cell_width + (cols - 1) * padding
combined_height = overlay_offset + rows * cell_height + (rows - 1) * padding

# 9. Create canvas and paste images
combined_image = Image.new("RGB", (combined_width, combined_height), bg_color)

for row_idx, row in enumerate(grid):
    for col_idx, view_name in enumerate(row):
        if view_name is None:
            continue

        img_path = os.path.join(step_file_dir, f"{view_name}.png")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        padded_img = resize_and_pad(img, img_size, bg_color)

        x = col_idx * (cell_width + padding)
        y = overlay_offset + row_idx * (cell_height + padding)

        combined_image.paste(padded_img, (x, y))
        img.close()  # Close the image file
        os.remove(img_path)  # Delete the image file from the disk

# 10. Draw text labels OVER the images
draw = ImageDraw.Draw(combined_image)

for row_idx, row in enumerate(grid):
    for col_idx, view_name in enumerate(row):
        if view_name is None:
            continue

        x = col_idx * (cell_width + padding)
        y = overlay_offset + row_idx * (cell_height + padding)

        title_text = view_name.replace("_", " ")
        bbox = font.getbbox(title_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = x + (cell_width - text_width) // 2 - 40
        text_y = y - 6  # Overlay slightly above top of image

        # Optional background box
        box_padding = 6
        box_coords = [
            text_x - box_padding,
            text_y - box_padding + 2,
            text_x + text_width + box_padding + 3 + 80,
            text_y + text_height + box_padding + 5
        ]
        draw.rectangle(box_coords, fill=(230, 230, 230))
        draw.text((text_x, text_y), "View: "+title_text, fill=(0, 0, 0), font=font)

# 11. Save result
output_path = os.path.splitext(step_file)[0] + ".png"
combined_image.save(output_path)
