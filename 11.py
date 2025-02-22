import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    return image, image_tensor

def detect_and_segment(model, image_tensor, confidence_threshold=0.5):
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    masks = prediction['masks'][prediction['scores'] > confidence_threshold]
    boxes = prediction['boxes'][prediction['scores'] > confidence_threshold]
    labels = prediction['labels'][prediction['scores'] > confidence_threshold]
    scores = prediction['scores'][prediction['scores'] > confidence_threshold]

    return masks, boxes, labels, scores

def visualize_results(image, masks, boxes, labels, scores):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    if len(masks) > 0:
        masks = masks.squeeze(1).detach().cpu().numpy()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))

        for i, (mask, box, label, score, color) in enumerate(zip(masks, boxes, labels, scores, colors)):
            plt.imshow(np.ma.masked_where(mask < 0.5, mask), alpha=0.5, cmap=plt.cm.hsv)
            x, y, w, h = box.detach().cpu().numpy()
            rect = plt.Rectangle((x, y), w-x, h-y, fill=False, edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x, y, f"{label.item()}: {score.item():.2f}", bbox=dict(facecolor=color, alpha=0.8), fontsize=8, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main(image_path):
    try:
        model = load_model()
        image, image_tensor = preprocess_image(image_path)
        masks, boxes, labels, scores = detect_and_segment(model, image_tensor)
        visualize_results(image, masks, boxes, labels, scores)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main('OI.jpeg')
