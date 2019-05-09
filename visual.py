import torch
from PIL import ImageDraw, ImageFont


def draw_rect(img, rects, labels=None):
    draw = ImageDraw.Draw(img)
    if isinstance(rects, torch.Tensor):
        rects = rects.tolist()
    for rect in rects:
        draw.rectangle(rect, outline='red', width=3)
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        font = ImageFont.truetype('calibri', 20)
        for rect, label in zip(rects, labels):
            xy = [(rect[0] + rect[2])/2, (rect[1] + rect[3])/2]
            draw.text(xy, str(label), font=font, fill='red', align='center')
    return img

