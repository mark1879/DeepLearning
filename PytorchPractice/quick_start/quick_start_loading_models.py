import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from quick_start import NeuralNetwork
import os

model = NeuralNetwork()
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
print(model_path)
model.load_state_dict(torch.load(model_path))

data_path = os.path.join(os.path.dirname(__file__), "data")  # 再加上它的相对路径，这样可以动态生成绝对路径
print(data_path)

training_data = datasets.FashionMNIST(
    root=data_path,
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root=data_path,
    train=False,
    download=True,
    transform=ToTensor(),
)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    print(pred[0].argmax(0))
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

