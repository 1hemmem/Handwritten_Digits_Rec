import utils
from PIL import Image
import torch
from torchvision import transforms

# Load the image
test_image = Image.open(
    "./model/English_Dataset/Img/img054-004.png"

)

# Setup the model
model = utils.setup("./model/ResNet18.pt")

# Set the model to evaluation mode
model.eval()

# Data reshaping
transform_pipeline = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),  # Resize if necessary for model input
        # transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
)

# Apply the transformations
test_image = transform_pipeline(test_image)

# Add batch dimension: shape will be [1, channels, height, width]
test_image = test_image.unsqueeze(0)

# Make the prediction
with torch.no_grad():  # Disable gradient computation for inference
    prediction = utils.predict(model,test_image)

prob, prediction = torch.max(prediction.flatten(), dim=0)
print(prediction)
print(prob)
# print(prediction)
