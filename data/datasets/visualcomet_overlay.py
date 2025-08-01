import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import shutil
from matplotlib.patches import Rectangle

def visualize_visualcomet_overlay(image_path, save_img_path, save_json_path, alpha=0.4):
    json_path = image_path.replace('.jpg', '.json')
    if not os.path.exists(json_path):
        return

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    boxes = json_data.get("boxes", [])
    names = json_data.get("names", [])
    width = json_data.get("width", 1920)
    height = json_data.get("height", 1080)

    person_count = sum(1 for name in names if name == "person")
    if person_count > 5:
        #########################
        if os.path.exists(save_img_path):
            os.makedirs(os.path.dirname(save_img_path.replace('visualcomet-bbox', 'visualcomet-bbox-big')), exist_ok=True)
            os.makedirs(os.path.dirname(save_json_path.replace('visualcomet-bbox', 'visualcomet-bbox-big')), exist_ok=True)
            shutil.move(save_img_path, save_img_path.replace('visualcomet-bbox', 'visualcomet-bbox-big'))
            shutil.move(save_json_path, save_json_path.replace('visualcomet-bbox', 'visualcomet-bbox-big'))
    return

    image = cv2.imread(image_path)
    if image is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(image)

    person_colors = [
        (255, 0, 0),       # 빨강
        (255, 128, 0),     # 주황
        (255, 255, 0),     # 노랑
        (0, 255, 0),       # 초록
        (0, 0, 255),       # 파랑
        (0, 128, 255),     # 남색
        (128, 0, 255),     # 보라
        (0, 0, 0),         # 검정
        (255, 255, 255),   # 흰색
    ]

    person_counter = 1
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box
        raw_label = names[i]

        if raw_label == "person":
            label = f"person {person_counter}"
            color_rgb = person_colors[(person_counter - 1) % len(person_colors)]
            color = np.array(color_rgb) / 255.0
            person_counter += 1
        else:
            label = raw_label
            color = np.array([128, 128, 128]) / 255.0

        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.tight_layout()

    # 저장 경로 생성
    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

    plt.savefig(save_img_path)
    plt.close()

    # JSON 파일도 같이 복사
    shutil.copyfile(json_path, save_json_path)


def process_visualcomet_directory(src_root, dst_root):
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if not file.endswith(".jpg"):
                continue

            image_path = os.path.join(root, file)
            rel_path = os.path.relpath(image_path, src_root)

            # 저장 경로 (jpg, json 모두)
            save_img_path = os.path.join(dst_root, rel_path)
            save_json_path = save_img_path.replace('.jpg', '.json')

            

            visualize_visualcomet_overlay(image_path, save_img_path, save_json_path)

src_dir = "/home/hice1/skim3513/scratch/VLA-VQA/datasets_src/images/visualcomet"
dst_dir = "/home/hice1/skim3513/scratch/VLA-VQA/datasets_src/images/visualcomet-bbox"

process_visualcomet_directory(src_dir, dst_dir)
