import torch
import torch.nn as nn 
import torchvision.transforms as transforms 
import io
from PIL import Image

# load model

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

inputSize = 784 #28x28
hiddenSize = 500
numClasses = 10
model = NeuralNet(inputSize, hiddenSize, numClasses)

PATH = './DL/mnist.pth'
model.load_state_dict(torch.load(PATH))
model.eval()

# image -> tensor
def transformImage(imageBytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081))])
        
    image = Image.open(io.BytesIO(imageBytes))
    return transform(image).unsqueeze(0)

# predict
def getPrediction(imageTensor):
    images = imageTensor.reshape(-1, 28*28)
    outputs = model(images)
    # max returns (values, index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
