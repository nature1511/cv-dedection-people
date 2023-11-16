import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#from utils.utils import SquarePad


class ImagesDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(
        self,
        path,
        device="cpu",
    ):
        super().__init__()
        self.device = device

        IMAGE_DIR = path
        jpg = list(IMAGE_DIR.rglob("*.jpg"))
        jpeg = list(IMAGE_DIR.rglob("*.jpeg"))
        png = list(IMAGE_DIR.rglob("*.png"))
        images = []
        images.extend(jpg)
        images.extend(jpeg)
        images.extend(png)

        self.images = images
        self._len = len(self.images)

        self.transform = transforms.Compose(
            [
                #SquarePad(),
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        """return torch.tensor(image) and shape original image"""
        file = self.images[index]
        tensor = Image.open(file).convert("RGB")
        tensor = self.transform(tensor)
        if self.device == "cuda" and torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor, file
