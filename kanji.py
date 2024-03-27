from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

app = Flask(__name__)


class ConvNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28 * 8, out_features=256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x


class RecognitionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=24 * 24 * 2, out_features=256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x


# Load your PyTorch model
model_path = 'model.pth'
model = torch.load('model.pth', map_location=torch.device('cpu'))
model.eval()

num_model = torch.load('number-model.pth', map_location=torch.device('cpu'))
num_model.eval()

# Transformation to convert PIL image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])
])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    # Receive image data from the frontend
    data_url = request.form['imageData']
    image_data = base64.b64decode(data_url.split(',')[1])

    inverted_image = transform_image(image_data)

    # Save the inverted image
    inverted_image.save('inverted_image.png')

    # Transform the grayscale image to a 2D tensor
    tensor_image = transform(inverted_image).unsqueeze(0)

    # Print the shape of the resulting tensor
    print(tensor_image.shape)
    print(tensor_image.max())

    # Get output
    # output = model(tensor_image)
    # _, predicted = torch.max(output, 1)

    output = num_model(tensor_image)
    _, predicted = torch.max(output, 1)

    corpus = ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち',
              'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め',
              'も', 'や', 'ゆ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'ゐ', 'ゑ', 'を', 'ん', 'ゝ']
    print(predicted.shape)
    print(corpus[predicted.item()])

    # Respond to the frontend
    response_data = {'message': 'Image received successfully!', 'letter': corpus[predicted.item()], 'number': predicted.item()}
    return jsonify(response_data)


def transform_image(image_data):
    img = Image.open(BytesIO(image_data))
    img_background = Image.new('RGB', (500, 500), 'white')
    img_background.paste(img, (0, 0), img)
    grayscale_img = img_background.convert('L')

    # left, top, right, bottom = find_roi(grayscale_img)
    # cropped_image = grayscale_img.crop((left, top, right, bottom))

    # inverted_image = ImageOps.invert(cropped_image)
    inverted_image = ImageOps.invert(grayscale_img)
    resized_img = inverted_image.resize((28, 28))

    return resized_img


def find_roi(image):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Find the bounding box around non-white pixels
    non_white_pixels = np.where(img_array != 255)  # Assuming white pixels have intensity 255
    top, left = np.min(non_white_pixels, axis=1)
    bottom, right = np.max(non_white_pixels, axis=1)

    return left, top, right, bottom


if __name__ == '__main__':
    app.run(debug=True)
