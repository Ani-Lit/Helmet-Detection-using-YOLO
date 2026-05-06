from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

class HelmetDetector:
    """Production-ready helmet detection using YOLOv8"""
    
    def __init__(self, model_path):
        """Initialize with model weights"""
        self.model = YOLO(model_path)
    
    def predict_image(self, image_path, conf=0.5):
        """Predict on single image"""
        results = self.model.predict(source=image_path, conf=conf)
        return results[0]
    
    def predict_batch(self, folder_path, conf=0.5):
        """Predict on multiple images"""
        results = self.model.predict(source=folder_path, conf=conf)
        return results
    
    def plot_result(self, result):
        """Display prediction with bounding boxes"""
        img = result.plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')
        plt.title('Helmet Detection Result')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    detector = HelmetDetector("weights/best.pt")
    result = detector.predict_image("test_image.jpg")
    detector.plot_result(result)
