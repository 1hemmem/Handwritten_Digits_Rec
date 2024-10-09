import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
import torch.nn.functional as F


class Cnn(nn.Module):
        def __init__(self, hidden_size, num_classes):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_classes = num_classes
            self.cnv1 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),stride=1)
            self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
            self.cnv2 = nn.Conv2d(in_channels=10,out_channels=50,kernel_size=(3,3),stride=1)
            self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
            
            self.fc1 = nn.Linear(1250,hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self,x):
            x = self.cnv1(x)
            x = F.relu(x)
            x = self.maxpool1(x)
            
            x = self.cnv2(x)
            x = F.relu(x)
            x = self.maxpool2(x)
            ## Flattening the input x
            x = torch.flatten(x,1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x
    

def setup(model_path):
#     model = torchvision.models.resnet18()
#     model.conv1 = nn.Conv2d(
#         1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#     )
    model = Cnn(1000,10)
    
    
    # model.features.conv0 = nn.Conv2d(
    #     in_channels=1,  # Change from 3 to 1 for grayscale images
    #     out_channels=model.features.conv0.out_channels,
    #     kernel_size=model.features.conv0.kernel_size,
    #     stride=model.features.conv0.stride,
    #     padding=model.features.conv0.padding,
    #     bias=False,
    # )
    # model = torchvision.models.resnet18()
    # model.conv1 = nn.Conv2d(
    #     1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    # )
    # # model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def transform(array):
    array = np.array(array, dtype=np.uint8)
    array = array.reshape((512, 512, 4))

    grayscale_array = array[:, :, 0]

    tensor_var = t(grayscale_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # tensor_var = tensor_var/255.0

    # tensor_var = 255 - tensor_var

    transform_pipeline = transforms.Compose(
        [
            # transforms.RandomRo0ation((-30,30)),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),  # Resize if necessary for model input
            # transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )
    tensor_var = transform_pipeline(tensor_var)
    print(tensor_var)
    # tensor_var = tensor_var/255.0
    # idx, val = torch.max(tensor_var)
    # print(idx)
    # print(val)
    print(tensor_var.shape)
    return tensor_var


def predict(model, image: torch.Tensor):
    model.eval()
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    return probs

def get_output_dict(probabilities: torch.Tensor):
    label_map = {str(i): i for i in range(10)}
    label_map.update({chr(65 + i): 10 + i for i in range(26)})
    label_map.update({chr(97 + i): 36 + i for i in range(26)})
    inv_map = {v: k for k, v in label_map.items()}

    prob, prediction = torch.max(probabilities.flatten(), dim=0)
    print(prediction)
    # prediction = inv_map[prediction.item()]

    return prob.item(), prediction.item()
