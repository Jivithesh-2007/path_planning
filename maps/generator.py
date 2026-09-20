"""
Sample map generator creating synthetic top-down floorplan images with start (green) and goal (red).
"""

from pathlib import Path
import numpy as np
import cv2


def generate_sample_map_image(output_path: Path | str = "maps/sample1.png") -> Path:
    """
    Generate a 600x600 top-down floorplan map image with walls, obstacles, rough terrain, start, and goal.

    :param output_path: Output PNG image path.
    :return: Path object of saved image.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 600x600 BGR canvas initialized to off-white background (240, 240, 245)
    img = np.full((600, 600, 3), 240, dtype=np.uint8)

    # Outer wall border (black, 15px thick)
    cv2.rectangle(img, (15, 15), (585, 585), (20, 20, 20), 15)

    # Internal wall obstacles (black shapes)
    cv2.rectangle(img, (120, 100), (160, 450), (20, 20, 20), -1)  # Vertical wall 1
    cv2.rectangle(img, (300, 150), (340, 500), (20, 20, 20), -1)  # Vertical wall 2
    cv2.rectangle(img, (420, 80), (460, 380), (20, 20, 20), -1)   # Vertical wall 3
    cv2.rectangle(img, (200, 280), (400, 310), (20, 20, 20), -1)  # Horizontal barrier

    # Rough terrain region (speckled textured patch)
    rough_patch = np.full((120, 120, 3), 150, dtype=np.uint8)
    noise = np.random.normal(0, 35, (120, 120, 3)).astype(np.int16)
    rough_patch = np.clip(rough_patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[180:300, 470:590] = rough_patch

    # Green Start Circle (OpenCV BGR: 0, 255, 0)
    start_center = (60, 60)
    cv2.circle(img, start_center, 18, (0, 255, 0), -1)
    cv2.circle(img, start_center, 18, (0, 100, 0), 2)

    # Red Goal Circle (OpenCV BGR: 0, 0, 255)
    goal_center = (540, 540)
    cv2.circle(img, goal_center, 18, (0, 0, 255), -1)
    cv2.circle(img, goal_center, 18, (100, 0, 0), 2)

    cv2.imwrite(str(path), img)
    print(f"Generated sample map image at {path}")
    return path


if __name__ == "__main__":
    generate_sample_map_image()
