# 🧠 Model Integration Documentation

## Overview
This dashboard uses a pre-trained Keras model for lip-sync-based deepfake detection. The model analyzes video chunks to classify them as REAL or FAKE based on lip movement patterns.

---

## 📦 Model File

### **Location**
```
deepfake_dashboard/models/cremad_model_finetuned_5600_5953.keras
```

### **Model Name**
`cremad_model_finetuned_5600_5953.keras`

This appears to be:
- **Base Dataset**: CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Fine-tuning**: Custom fine-tuning with 5600-5953 parameters or epochs
- **Format**: Keras/TensorFlow SavedModel (.keras extension)

### **Expected Architecture**
Based on the code implementation, the model is likely:
- **Type**: CNN + BiLSTM (Convolutional Neural Network + Bidirectional LSTM)
- **Purpose**: Temporal sequence analysis of lip regions
- **Output**: Binary classification (REAL vs FAKE)

---

## 🔄 How the Model is Used

### **1. Model Loading**
```python
def load_model(model_path: str):
    global keras
    if keras is None:
        import tensorflow as tf
        from tensorflow import keras as _keras
        keras = _keras
    model = keras.models.load_model(model_path)
    return model
```

**Location**: Line 40-47 in `app.py`

**Process**:
- Lazy imports TensorFlow/Keras (only when needed)
- Uses `keras.models.load_model()` to load the `.keras` file
- Includes error handling with a fallback dummy model

---

### **2. Input Preprocessing**

#### **Video Processing Pipeline**

**Step 1: Lip Region Extraction**
```python
# MediaPipe Face Mesh detects facial landmarks
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Extract lip region using landmarks 61-88 (lip contour)
LIPS_IDX = set(list(range(61, 88)) + [0, 13, 14, 17])
```

**Step 2: Frame Chunking**
```python
# Video is divided into chunks of 15 frames (~1 second at 15fps)
CHUNK_SIZE_FRAMES = 15

# Each frame:
# - Lip ROI extracted and resized to 128x128
# - Converted to grayscale
# - Normalized to [0, 1] range
```

**Step 3: Input Tensor Creation**
```python
# Per chunk: (15, 128, 128) → normalized → (15, 128, 128, 1)
# Final shape: (N, T, H, W, C)
X_frames = np.stack(X_frames, axis=0)  # Shape: (N, 15, 128, 128, 1)
```

Where:
- **N** = Number of chunks
- **T** = Temporal frames (15)
- **H** = Height (128)
- **W** = Width (128)
- **C** = Channels (1 for grayscale)

#### **Audio Features (Optional)**
```python
# MFCC (Mel-frequency cepstral coefficients) extraction
mfcc = librosa.feature.mfcc(y=audio_segment, sr=16000, n_mfcc=20)
# Shape per chunk: (20, 44)
```

**Note**: Audio MFCCs are computed but currently not fed to the model in the prediction step. They're available for models that accept multimodal input.

---

### **3. Model Inference**

```python
# Primary prediction attempt
preds = model.predict(X_frames, verbose=0)
# Input shape: (N, 15, 128, 128, 1)
# Output shape: (N, 1) or (N,) - probability scores

# Fallback for different input requirements
if prediction fails:
    X_alt = X_frames.reshape((N, 15, 16384))  # Flatten spatial dims
    preds = model.predict(X_alt, verbose=0)
```

**Adaptive Input Handling**:
1. **First attempt**: Full 4D tensor `(N, 15, 128, 128, 1)`
2. **Second attempt**: Flattened spatial `(N, 15, 16384)` for models expecting flattened frames
3. **Fallback**: Dummy predictions (0.5) if both fail

---

### **4. Output Processing**

```python
# Raw predictions (probability scores)
preds = preds.squeeze()  # Remove extra dimensions

# Classification logic
for confidence_score in preds:
    label = "REAL" if confidence_score >= 0.5 else "FAKE"
```

**Interpretation**:
- **Score ≥ 0.5** → REAL video
- **Score < 0.5** → FAKE video (deepfake detected)

**Per-Chunk Results**:
```json
{
  "chunk": 0,
  "start": 0.0,
  "end": 1.0,
  "confidence": 0.7234,
  "label": "REAL",
  "keyframe": "static/frames/abc123_chunk0000.jpg"
}
```

**Final Verdict**:
```python
# Majority voting
real_count = sum(1 for pred in predictions if pred >= 0.5)
fake_count = len(predictions) - real_count
final_label = "REAL" if real_count >= fake_count else "FAKE"
```

---

## 🎯 Model Requirements

### **Expected Input Shape**
The model should accept one of these formats:

**Option 1: 5D Tensor (Recommended)**
```
Shape: (batch_size, timesteps, height, width, channels)
Example: (32, 15, 128, 128, 1)
```

**Option 2: 3D Tensor (Flattened)**
```
Shape: (batch_size, timesteps, features)
Example: (32, 15, 16384)
```

### **Expected Output Shape**
```
Shape: (batch_size, 1) or (batch_size,)
Type: Float32
Range: [0.0, 1.0] (probability)
```

---

## 🔧 How to Use Your Own Model

### **Step 1: Save Your Model**
```python
# Save in Keras format
model.save('models/your_model_name.keras')
```

### **Step 2: Update Model Path**
In `app.py`, line 60:
```python
model_path = os.path.join(app_config["MODELS_FOLDER"], "your_model_name.keras")
```

### **Step 3: Verify Input Compatibility**
Ensure your model accepts:
- Input shape: `(None, 15, 128, 128, 1)` or `(None, 15, 16384)`
- Output: Single probability score per sample

### **Step 4: (Optional) Modify Preprocessing**
If your model needs different preprocessing:

**Change frame size**:
```python
# Line 168 in app.py
lip_roi = cv2.resize(lip_roi, (your_width, your_height))
```

**Change chunk size**:
```python
# Line 20 in app.py
CHUNK_SIZE_FRAMES = your_chunk_size
```

**Add audio input**:
```python
# After line 257, modify prediction to include audio
preds = model.predict([X_frames, mfcc_chunks], verbose=0)
```

---

## 🛡️ Fallback Mechanism

### **Dummy Model**
If the actual model file is missing or fails to load:

```python
class DummyModel:
    def predict(self, x, verbose=0):
        n = x.shape[0]
        return np.full((n, 1), 0.5, dtype=np.float32)
```

**Behavior**:
- Returns 0.5 confidence for all chunks
- All chunks classified as REAL (since 0.5 ≥ 0.5)
- Allows UI testing without the actual model
- `model_loaded: false` flag in JSON output

---

## 📊 Model Performance Metrics

The dashboard **does not** currently display:
- ❌ Accuracy, precision, recall
- ❌ ROC curves or confusion matrices
- ❌ Model training history

**What it does show**:
- ✅ Per-chunk confidence scores
- ✅ Temporal classification timeline
- ✅ Frame-by-frame analysis
- ✅ Audio waveform overlay
- ✅ Final aggregated verdict

---

## 🚀 Example Model Architectures

### **Compatible CNN+BiLSTM Example**
```python
from tensorflow.keras import layers, models

def create_model():
    inputs = layers.Input(shape=(15, 128, 128, 1))
    
    # TimeDistributed CNN for feature extraction
    x = layers.TimeDistributed(layers.Conv2D(32, 3, activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 3, activation='relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    # BiLSTM for temporal modeling
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.save('models/cremad_model_finetuned_5600_5953.keras')
```

---

## 📝 Summary

| Aspect | Details |
|--------|---------|
| **Model File** | `models/cremad_model_finetuned_5600_5953.keras` |
| **Framework** | TensorFlow/Keras |
| **Input Type** | Video lip regions (grayscale frames) |
| **Input Shape** | `(N, 15, 128, 128, 1)` or `(N, 15, 16384)` |
| **Output Type** | Binary classification probability |
| **Output Shape** | `(N, 1)` or `(N,)` |
| **Threshold** | 0.5 (REAL if ≥ 0.5, FAKE if < 0.5) |
| **Chunk Size** | 15 frames (~1 second) |
| **Processing** | MediaPipe lip extraction + normalization |
| **Fallback** | Dummy model with 0.5 predictions |

---

## 🎓 For Your Final Year Project

**To use your actual trained model**:
1. Place your `.keras` model file in `deepfake_dashboard/models/`
2. Update the filename in `app.py` line 60
3. Ensure your model input/output shapes match the specification
4. Test with a sample video to verify predictions

**Current model file is a placeholder** - replace it with your actual trained deepfake detection model!
