"""
This file builds an ASL reference grid for testing purposes that will be displayed
along the webcam in main.py, this will allow for easier testing as a non ASL user.
"""

import math
from asl import paths
import numpy as np
import cv2

def first_clean_image_per_letter():
    d = np.load(paths.CLEAN_FEATURES)
    y, img_paths = d["y"], d["paths"]

    return {str(c): img_paths[y == c][0] for c in np.unique(y)}

def build_reference_grid(ref_path, cell=110, cols=7):
    letters = sorted(ref_path)
    rows = math.ceil(len(letters) / cols)
    grid = np.zeros((rows * cell, cols * cell, 3), np.uint8) # stores the pixel values of the images
    cell_pos = {}

    for i, letter in enumerate(letters):
        path = paths.ROOT / ref_path[letter]
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"Could not read reference image: {path}")
        img = cv2.resize(img, (cell, cell))
        cv2.putText(img, letter, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # putting in grid
        row, col = divmod(i, cols)
        y0, x0 = row * cell, col * cell
        grid[y0: y0 + cell, x0: x0 + cell] = img
        cell_pos[letter] = (x0, y0)

    return grid, cell_pos


if __name__ == "__main__":
    grid, _ = build_reference_grid(first_clean_image_per_letter())
    print("grid shape:", grid.shape)            
    cv2.imshow("Reference", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
