import torch
from torchvision import models
from PIL import Image
from backend.utils import get_imagenet_class
import os
import shutil
from tqdm import tqdm

class std_clf:
    def __init__(self, model_type, preprocessor):
        self.model_type = model_type
        self.model = self._get_backend(self.model_type)
        self.model.eval()
        self.preprocessor = preprocessor

    def _get_backend(self, model_type):
        if model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
        elif model_type == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)
        else:
            model = models.resnet152(pretrained=True)
        return model

    def process_images(self, files, out_path):
        print('Processing images: ')
        with torch.no_grad():
            for f in tqdm(files):
                img = Image.open(f)
                img = self.preprocessor(img).unsqueeze(0)
                logits = self.model(img)
                pred_class = get_imagenet_class(logits)
                
                save_path = os.path.join(out_path, pred_class)
                if not os.path.exists(save_path): 
                    os.makedirs(save_path)
                shutil.copy(f, os.path.join(save_path, f.split('/')[-1]))
