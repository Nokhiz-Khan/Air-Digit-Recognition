
# Air Digit Recognition ✍️

Write digits in thin air using just your index finger — no touchscreen, no mouse. A webcam tracks your finger, draws what you write, and a trained neural network reads it back to you in real time.

## How It Works

1. **MediaPipe Hands** detects your hand and tracks the index finger tip in every webcam frame.
2. **OpenCV** draws a trail on a virtual canvas as your finger moves — but only while your index finger is extended and the other fingers are folded, giving natural "pen up / pen down" control.
3. When you press **'p'**, the drawn trail is cropped to its bounding box, padded, and resized to 28x28 pixels to match the MNIST format.
4. A **CNN trained on the MNIST dataset** predicts the digit and displays the result along with a confidence score.

## Demo

Screenshots of the app in action are available in the [`images/`](images) folder.

## Controls

| Key | Action |
|---|---|
| Point with index finger (fold others) | Draw |
| `c` | Clear the canvas |
| `p` | Predict the drawn digit |
| `q` | Quit |

## Tech Stack

- **Python**
- **OpenCV** — webcam capture and drawing
- **MediaPipe** — real-time hand landmark detection
- **TensorFlow / Keras** — CNN model trained on MNIST

## Project Structure

```
main.py                      - Run this: real-time app
requirements.txt
digit_weights.weights.h5     - Pre-trained model weights
training/
    train_model.ipynb        - Notebook used to train the CNN on MNIST
images/                      - Demo screenshots
```


## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Why This Project

Most digit-recognition projects stop at a Jupyter notebook demo. This one goes a step further by building a full real-time input pipeline — hand tracking, gesture-based drawing control, and live inference — combining classical computer vision (finger tracking, image preprocessing) with a trained deep learning model.
