
from PIL import Image, ImageDraw, ImageEnhance
import random
import os
import math

from templateconfig.t2config import bubbles


def random_gray_color():
    """随机生成灰度颜色"""
    gray = random.choices(
        [
            random.randint(5, 30),      # 黑色
            random.randint(31, 65),     # 深灰
            random.randint(66, 110),    # 灰色
            random.randint(111, 160),   # 浅灰
        ],
        weights=[20, 40, 30, 10],
        k=1
    )[0]

    return (gray, gray, gray)



def random_fill_bubble(img, x, y, r):
    """
    随机填涂气泡：
    1. 随机灰度颜色
    2. 小幅随机放缩
    3. 随机椭圆形状
    4. 小幅随机偏移
    5. 随机笔迹
    """
    draw = ImageDraw.Draw(img)

    # 随机填涂程度
    fill_type = random.choices(
        ["light", "normal", "heavy"],
        weights=[20, 55, 25],
        k=1
    )[0]

    if fill_type == "light":
        size_ratio = random.uniform(0.70, 0.80)
        stroke_count = random.randint(1, 3)

    elif fill_type == "normal":
        size_ratio = random.uniform(0.80, 0.90)
        stroke_count = random.randint(3, 7)

    else:
        size_ratio = random.uniform(0.90, 1.00)
        stroke_count = random.randint(6, 12)

    # 整体随机放缩（小幅度）
    scale = random.uniform(0.95, 1.05)

    # 随机中心偏移
    cx = x + random.uniform(-1, 1)
    cy = y + random.uniform(-1, 1)

    # 随机长宽比例
    rx = r * size_ratio * scale * random.uniform(0.95, 1.05)
    ry = r * size_ratio * scale * random.uniform(0.95, 1.05)

    # 随机边缘不规则程度
    points = []

    for i in range(48):
        angle = 2 * math.pi * i / 48
        noise = random.uniform(0.95, 1.05)

        px = cx + rx * noise * math.cos(angle)
        py = cy + ry * noise * math.sin(angle)

        points.append((px, py))

    # 填涂主体
    draw.polygon(
        points,
        fill=random_gray_color()
    )

    # 随机叠加笔迹
    for _ in range(stroke_count):
        sx = cx + random.uniform(-rx * 0.65, rx * 0.65)
        sy = cy + random.uniform(-ry * 0.65, ry * 0.65)

        ex = sx + random.uniform(-r * 0.25, r * 0.25)
        ey = sy + random.uniform(-r * 0.25, r * 0.25)

        draw.line(
            [(sx, sy), (ex, ey)],
            fill=random_gray_color(),
            width=random.randint(1, 2)
        )

    # 少量随机小笔迹
    for _ in range(random.randint(0, 2)):
        sx = cx + random.uniform(-rx * 0.8, rx * 0.8)
        sy = cy + random.uniform(-ry * 0.8, ry * 0.8)

        size = random.uniform(1, 2)

        draw.ellipse(
            [
                sx - size,
                sy - size,
                sx + size,
                sy + size
            ],
            fill=random_gray_color()
        )


def generate_one(
    template_path,
    save_img_path,
    save_label_path,
    img_w,
    img_h
):
    img = Image.open(template_path).convert("RGB")

    label_lines = []

    for q_num, options in bubbles.items():

        # 随机选择一个选项，或者空选
        choices = list(options.keys()) + [None]
        selected = random.choice(choices)

        for opt, (x, y, r) in options.items():

            is_filled = (opt == selected)

            if is_filled:
                random_fill_bubble(img, x, y, r)

            # YOLO 标签
            class_id = 0 if is_filled else 1

            cx = x / img_w
            cy = y / img_h

            w = (2 * r) / img_w
            h = (2 * r) / img_h

            label_lines.append(
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{w:.6f} {h:.6f}"
            )

    # 随机旋转
    if random.random() < 0.5:
        angle = random.uniform(-1.5, 1.5)

        img = img.rotate(
            angle,
            expand=False,
            fillcolor=(255, 255, 255)
        )

    # 随机亮度
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(
        random.uniform(0.90, 1.10)
    )

    # 保存图片
    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_label_path), exist_ok=True)

    img.save(save_img_path, quality=95)

    # 保存标签
    with open(save_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))


def batch_g(template_path):

    img_w, img_h = 1043, 1418

    GENERATE_NUM = 100

    for i in range(GENERATE_NUM):

        img_path = f"dataset/images/train/{i:04d}.jpg"
        label_path = f"dataset/labels/train/{i:04d}.txt"

        generate_one(
            template_path,
            img_path,
            label_path,
            img_w,
            img_h
        )

        print(f"生成 {i + 1}/{GENERATE_NUM}")


if __name__ == "__main__":

    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)

    batch_g("./template/t2.jpg")
