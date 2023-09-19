import torch
from torchvision.models import resnet50
import torchvision.transforms as transforms


class AgePredictor:
    def __init__(self, root_dir, device):
        self.root_dir = root_dir
        self.device = device
        _model_cv = resnet50()
        _model_cv.fc = torch.nn.Linear(in_features=_model_cv.fc.in_features,
                                        out_features=1) 
        _model_cv.eval()
        _model_cv.to(self.device)
        _model_cv.load_state_dict(torch.load(f"{self.root_dir}/model_cv_age.pth", 
                                    map_location=self.device))
        self.model_cv = _model_cv
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

    def transform_image(self, image):
        image = self.transform(image).unsqueeze(0)
        return image

    def get_age_by_image(self, image):
        with torch.no_grad():
            output = self.model_cv(image.to(self.device))
        age = int(output.item())
        return age

