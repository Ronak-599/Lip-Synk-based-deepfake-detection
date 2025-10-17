# 🔄 Model Workflow Summary

## Quick Reference: How the Model is Used

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO UPLOAD (MP4/AVI/etc.)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Extract Frames (OpenCV)                                     │
│  2. Detect Face Landmarks (MediaPipe Face Mesh)                 │
│  3. Crop Lip Region (landmarks 61-88)                           │
│  4. Resize to 128x128 grayscale                                 │
│  5. Group into chunks of 15 frames                              │
│  6. Normalize to [0, 1]                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT TENSOR CREATION                         │
├─────────────────────────────────────────────────────────────────┤
│  Shape: (N, 15, 128, 128, 1)                                   │
│                                                                 │
│  Where:                                                         │
│    N = Number of chunks                                         │
│    15 = Frames per chunk (~1 second)                            │
│    128 = Width/Height of lip image                              │
│    1 = Grayscale channel                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL: cremad_model_finetuned_5600_5953.keras      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Architecture (Expected):                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  TimeDistributed CNN Layers                          │      │
│  │    ↓ (Extract spatial features from each frame)      │      │
│  │  Bidirectional LSTM Layers                           │      │
│  │    ↓ (Model temporal dependencies)                   │      │
│  │  Dense + Sigmoid Output                              │      │
│  │    ↓ (Binary classification: REAL vs FAKE)           │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  Input:  (N, 15, 128, 128, 1)                                  │
│  Output: (N, 1) - probability scores                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION & CLASSIFICATION                   │
├─────────────────────────────────────────────────────────────────┤
│  For each chunk:                                                │
│    confidence_score = model.predict(chunk)                      │
│                                                                 │
│    if confidence_score >= 0.5:                                  │
│        label = "REAL"                                           │
│    else:                                                        │
│        label = "FAKE"                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGGREGATION & RESULTS                         │
├─────────────────────────────────────────────────────────────────┤
│  Per-chunk results:                                             │
│    • Time range (start, end)                                    │
│    • Confidence score                                           │
│    • Label (REAL/FAKE)                                          │
│    • Keyframe image                                             │
│                                                                 │
│  Final verdict:                                                 │
│    • Majority voting across all chunks                          │
│    • Average confidence                                         │
│    • Processing time                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DASHBOARD VISUALIZATION                       │
├─────────────────────────────────────────────────────────────────┤
│  • Video player with color-coded timeline                       │
│  • Chunk-wise analysis table                                    │
│  • Confidence line chart                                        │
│  • Extracted lip frames grid                                    │
│  • Audio waveform with overlays                                 │
│  • Final summary card                                           │
│  • CSV report download                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model File Location

```
📁 deepfake_dashboard/
    📁 models/
        📄 cremad_model_finetuned_5600_5953.keras  ← YOUR MODEL HERE
```

**Status**: ⚠️ Currently using fallback dummy model (outputs 0.5 for all chunks)

---

## To Use Your Trained Model

### Step 1: Place Model File
```bash
# Copy your trained model to:
deepfake_dashboard/models/cremad_model_finetuned_5600_5953.keras

# Or rename your model and update app.py line 60
```

### Step 2: Verify Model Compatibility
Your model should:
- ✅ Accept input shape: `(None, 15, 128, 128, 1)`
- ✅ Output shape: `(None, 1)` or `(None,)`
- ✅ Output range: 0.0 to 1.0 (sigmoid activation)
- ✅ Be saved in Keras format (`.keras` or `.h5`)

### Step 3: Test
```bash
python app.py
# Upload a test video
# Check if predictions are reasonable (not all 0.5)
```

---

## Model Input Details

### Frame Processing
```python
# Each video frame undergoes:
1. Face detection (MediaPipe)
2. Lip landmark extraction (61-88)
3. Bounding box expansion (1.6x)
4. Crop & resize to 128x128
5. Convert to grayscale
6. Normalize to [0, 1]
```

### Chunk Creation
```python
# Frames grouped into temporal sequences:
- Chunk size: 15 frames
- ~1 second duration at 15fps
- Shape per chunk: (15, 128, 128, 1)
```

### Batch Processing
```python
# Multiple chunks processed together:
X_frames = np.stack(chunks, axis=0)
# Final shape: (N_chunks, 15, 128, 128, 1)

# Model prediction:
predictions = model.predict(X_frames)
# Output: (N_chunks, 1) probability scores
```

---

## Classification Logic

```python
# Per chunk:
if prediction >= 0.5:
    label = "REAL"     # Authentic video
else:
    label = "FAKE"     # Deepfake detected

# Final verdict (majority voting):
real_chunks = count(predictions >= 0.5)
fake_chunks = count(predictions < 0.5)

if real_chunks >= fake_chunks:
    final_verdict = "REAL"
else:
    final_verdict = "FAKE"
```

---

## Alternative: Use Your Own Model Architecture

If your model has a different architecture, modify these sections in `app.py`:

### 1. Input Preprocessing (Lines 150-200)
```python
# Change image size
lip_roi = cv2.resize(lip_roi, (YOUR_WIDTH, YOUR_HEIGHT))

# Change chunk size
CHUNK_SIZE_FRAMES = YOUR_CHUNK_SIZE
```

### 2. Model Input Format (Lines 195-210)
```python
# Example: RGB instead of grayscale
lip_roi_rgb = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2RGB)
frames_current_chunk.append(lip_roi_rgb)

# Later:
arr = np.expand_dims(arr, -1)  # Remove this for RGB (already 3 channels)
```

### 3. Multimodal Input (Lines 255-265)
```python
# If your model uses both video + audio:
preds = model.predict([X_frames, mfcc_chunks], verbose=0)
```

---

## Current Model: CREMA-D Based

**CREMA-D** = Crowd-sourced Emotional Multimodal Actors Dataset
- Originally for emotion recognition
- Fine-tuned for deepfake detection
- File suffix `5600_5953` likely indicates:
  - Training epochs, or
  - Parameter count, or
  - Dataset size

**Your model**: Replace with your actual trained model for real predictions!

---

## Questions?

1. **Where is the model loaded?** → `app.py`, line 78-90
2. **What if model is missing?** → Dummy model with 0.5 predictions (line 85-92)
3. **How to change threshold?** → Line 272: `label = "REAL" if p >= 0.5 else "FAKE"`
4. **How to use audio?** → MFCCs computed but not used; add to prediction call
5. **Model format?** → Keras `.keras` or `.h5` format

---

See `MODEL_DETAILS.md` for complete technical documentation!
