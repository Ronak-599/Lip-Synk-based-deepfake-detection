# 🎯 Lip-Sync-Based Deepfake Detection Dashboard

A Flask-based web dashboard to analyze videos using a pre-trained Keras model and visualize chunk-wise deepfake predictions with timelines, charts, frames, and audio overlays.

## 📚 Documentation
- **[MODEL_DETAILS.md](MODEL_DETAILS.md)** - Complete model integration guide
- **[MODEL_USAGE.md](MODEL_USAGE.md)** - Quick reference and workflow diagram
- **[README.md](README.md)** - Setup and installation (this file)

## Features
- Upload video (.mp4/.avi/.flv/.mov/.mkv)
- Background analysis with progress polling
- MediaPipe lip ROI extraction, 15-frame chunks (~1s)
- Audio extraction (MoviePy) + MFCC (librosa)
- Model inference (CNN+BiLSTM-ready) with fallbacks
- Dashboard with:
  - Video + color-coded timeline
  - Chunk table with keyframes
  - Chart.js confidence line
  - Lip frame grid
  - Plotly audio overlay with REAL/FAKE regions
  - Final summary + CSV download

## Project Structure
```
/deepfake_dashboard
├── app.py                      # Flask backend & processing pipeline
├── models/
│   └── cremad_model_finetuned_5600_5953.keras  # ⚠️ PLACE YOUR MODEL HERE
├── static/
│   ├── uploads/                # Uploaded videos (auto-created)
│   ├── frames/                 # Extracted lip frames (auto-created)
│   ├── results/                # JSON results & audio (auto-created)
│   ├── css/style.css          # Modern UI styling
│   └── js/dashboard.js        # Interactive visualizations
├── templates/
│   ├── index.html             # Upload page
│   └── result.html            # Results dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # Setup guide (this file)
├── MODEL_DETAILS.md           # Complete model documentation
└── MODEL_USAGE.md             # Quick reference & workflow
```

## 🧠 Model Information

**Model File**: `models/cremad_model_finetuned_5600_5953.keras`

**Current Status**: ⚠️ Using fallback dummy model (outputs 0.5)

**To use your trained model**:
1. Place your `.keras` model file in `models/` folder
2. Update filename in `app.py` line 60 if different
3. Ensure model accepts input shape: `(None, 15, 128, 128, 1)`
4. Model should output: `(None, 1)` probability scores

**See [MODEL_DETAILS.md](MODEL_DETAILS.md) for complete integration guide**

## Prerequisites (Windows)
- Python 3.10 or 3.11 recommended
- FFmpeg installed and on PATH (required by MoviePy)
- Visual C++ Build Tools (for some packages) if needed

## Setup
```powershell
# 1) Create and activate venv
python -m venv .venv
. .venv\\Scripts\\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Place your model file
# Copy cremad_model_finetuned_5600_5953.keras into models/ folder

# 4) Run the app
python app.py
```

Then open http://localhost:5000 in your browser.

## Notes
- If TensorFlow GPU is desired, install the appropriate tensorflow package + CUDA/CuDNN.
- For large videos, processing can take time. Progress updates will show in the result page.
- If your model expects a specific input shape, adjust the preprocessing/inference logic in `background_analyze`.

## Troubleshooting
- If MoviePy errors about audio, ensure FFmpeg is installed.
- If librosa complains about soundfile, ensure `soundfile` is installed and the audio codec is supported.
- For CORS / static file access, the app serves local paths under `/static` by default.
