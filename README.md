
# ✨ Sign Language Recognition System (LSTM + MediaPipe)

This project is a real-time sign language recognition system that uses a webcam to detect and classify hand and body gestures into predefined phrases such as "hi how are you" and "thank you so much". It leverages MediaPipe for landmark detection (face, pose, and hands) and a Long Short-Term Memory (LSTM) neural network to learn and predict gesture sequences over time.

Key features include:

-Real-time gesture capture using OpenCV and MediaPipe

-Sequence classification using an LSTM-based neural network

-Evaluation with classification report and confusion matrix

-Live webcam inference and overlay of predicted text

-Modular and extensible for more gesture classes

---

## 📁 Project Structure

```
MP_Data/                 # Preprocessed gesture data (npy files)
├── hi how are you/
├── thank you so much/
Logs/                    # TensorBoard logs
main.py                  # This code file
requirements.txt
README.md
```

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Prepare Data:**

   Ensure your directory follows this format:
   ```
   MP_Data/
   ├── hi how are you/
   │   ├── 0/
   │   │   ├── 0.npy
   │   │   ├── ...
   ├── thank you so much/
   ```

2. **Train the Model:**

   Run the script to train:

   ```bash
   python main.py
   ```

3. **Real-time Inference:**

   After training, the webcam feed will start. Perform gestures in front of your camera.

   Press `q` to quit.

---

## 🧠 Model Details

- **Architecture**: 3 LSTM layers + Dense layers
- **Input shape**: `(30, 1662)` — 30 frames, 1662 keypoints
- **Output**: Probability scores for each action

---

## 📊 Evaluation

After training:
- **Classification report** and **confusion matrix** are printed.
- Visualization shows real vs predicted gesture labels.

---

## 🖼️ Screenshots
- 📊Model Accuracy and loss during training
![WhatsApp Image 2025-03-29 at 17 51 24_c9d3a8d4](https://github.com/user-attachments/assets/26a8cd9e-9b4e-44aa-b4b7-275f1c4eb4a9)

