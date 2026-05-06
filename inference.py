from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

class HelmetDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image_path, conf=0.5):
        return self.model.predict(source=image_path, conf=conf)[0]
    
    def plot(self, result):
        img = result.plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')
        plt.title('Helmet Detection')
        plt.show()
