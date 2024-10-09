import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np


def setup(model_path):
    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def transform(array):
    array = np.array(array, dtype=np.uint8)
    array = array.reshape((512, 512, 4))

    grayscale_array = array[:, :, 0]

    tensor_var = t(grayscale_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    transform_pipeline = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),  # Resize if necessary for model input
        ]
    )
    tensor_var = transform_pipeline(tensor_var)
    return tensor_var


def predict(model, image: torch.Tensor):
    model.eval()
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    return probs


def get_output_dict(probabilities: torch.Tensor):
    prob, prediction = torch.max(probabilities.flatten(), dim=0)
    print(prediction)

    return prob.item(), prediction.item()
