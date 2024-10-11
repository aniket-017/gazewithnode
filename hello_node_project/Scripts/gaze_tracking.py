import torch
import torchvision.models as models
from torch import nn
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import os
import requests  # Import requests to check for the stop signal

# Define the custom gaze model based on VGG16
class GazeModel(nn.Module):
    def __init__(self):
        super(GazeModel, self).__init__()
        # Load a pre-trained VGG16 model
        self.vgg16 = models.vgg16()
        # Modify the classifier for gaze estimation
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, 2)  # Output layer has 2 neurons for (x, y)

    def forward(self, x):
        return self.vgg16(x)

# Load the pretrained gaze model
model = GazeModel()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gaze_model_pytorch_vgg16_prl_mpii_allsubjects4.model')

# Load the state dictionary with adjustments if needed
try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    new_state_dict = {}

    for k, v in state_dict.items():
        new_key = k.replace("left_features", "vgg16.features")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()  # Set the model to evaluation mode

# Define transformation for input frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to save gaze data to CSV
def save_to_csv(gaze_data, filename="gaze_data.csv"):
    if gaze_data:
        df = pd.DataFrame(gaze_data, columns=['x', 'y'])
        df.to_csv(filename, index=False)
        print(f"Gaze data saved to {filename}")
    else:
        print("No gaze data to save.")

# Function to generate heatmap
def generate_heatmap(gaze_data, filename="heatmap.png"):
    if gaze_data:
        df = pd.DataFrame(gaze_data, columns=['x', 'y'])
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='x', y='y', cmap="Reds", fill=True, bw_adjust=0.5)
        plt.title('Gaze Heatmap')
        plt.gca().invert_yaxis()
        plt.savefig(filename)
        plt.close()
        print(f"Heatmap saved to {filename}")
    else:
        print("No gaze data available to generate a heatmap.")

# Path to the image file that updates every 10ms
image_path = os.path.join(os.path.dirname(__file__), 'current_frame.jpg') # Change this to the actual path of the current_frame.jpg

gaze_data = deque()

# URL to check stop signal
stop_url = "http://localhost:3000/stop_signal"

# Main loop to process the image and predict gaze location
while True:
    # Load the current frame image
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not load current frame image.")
            continue

        # Convert frame to PIL image and apply transformations
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0)

        # Predict gaze using the model
        with torch.no_grad():
            predicted_gaze = model(input_tensor)

        # Extract gaze coordinates (x, y)
        gaze_coords = predicted_gaze.numpy()[0]

        # Print predicted gaze coordinates
        print(f"Predicted gaze: {gaze_coords}")

        # Append gaze coordinates to the deque
        gaze_data.append((gaze_coords[0], gaze_coords[1]))

        # Optional: Draw gaze point on the frame (visual feedback)
        cv2.circle(frame, (int(gaze_coords[0]), int(gaze_coords[1])), 5, (0, 255, 0), -1)

        # Display the frame (optional)
        cv2.imshow('Gaze Tracking', frame)

        # Check stop signal from server
        try:
            response = requests.get(stop_url)
            if response.text == 'STOP':
                print("Stopping the program as stop signal received")
                break
        except Exception as e:
            print(f"Error checking stop signal: {e}")

        # Add small delay to simulate the update frequency (10ms)
        cv2.waitKey(10)

    except Exception as e:
        print(f"Error processing frame: {e}")
        break

# Save data and clean up
cv2.destroyAllWindows()
save_to_csv(gaze_data)
generate_heatmap(gaze_data)
