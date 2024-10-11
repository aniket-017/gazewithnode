
# Gaze Estimation and Tracking Using VGG16

This project implements a gaze tracking model using a pre-trained VGG16 model. The model predicts where a user is looking on the screen based on webcam footage. The gaze estimation uses deep learning and linear regression to map predicted gaze coordinates to screen positions.

## Project Structure
- **GazeModel**: A custom PyTorch model built on VGG16, modified for gaze tracking.
- **Gaze Data Collection**: Uses webcam feed to capture frames and predict the user's gaze coordinates.
- **Calibration**: A calibration phase that aligns gaze prediction with actual screen coordinates.
- **Data Export**: Gaze data can be saved to CSV, and heatmaps can be generated using seaborn.

## Setup

### Step 1: Virtual Environment
Create a virtual environment for the project:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
```bash
venv\Scripts\activate
```

- On macOS/Linux:
```bash
source venv/bin/activate
```

### Step 2: Install Dependencies
Install the required packages:

```bash
pip install torch torchvision opencv-python pillow numpy pandas matplotlib seaborn scikit-learn pyautogui
```

### Step 3: Download Pretrained Model
Download the pretrained model from the following Google Drive link:

[Pretrained Gaze Model](https://drive.google.com/file/d/1hAe6cwEdOOPuzvQ_eKIRIEA-vvEJqwZb/view?usp=drive_link)

Place the downloaded model in the correct path as specified in the code (`model_path`).

### Step 4: Running the Project
Once the model is downloaded and the environment is set up, you can run the project using:

```bash
python gaze_tracking_project.py
```

## Usage
- The model will open your webcam and start tracking your gaze.
- Look at the red dots on the screen for calibration.
- Once calibrated, your gaze coordinates will be displayed.
- Press `q` to stop the tracking and exit the program.

## Output
- Gaze data is saved in a CSV file.
- A heatmap is generated to visualize gaze patterns.

