"""
Quick test script to verify model loading and inference
"""
import os
import numpy as np

print("=" * 60)
print("Testing Model Loading and Inference")
print("=" * 60)

# Test 1: Check if model file exists
model_path = "models/cremad_model_finetuned_5600_5953.keras"
print(f"\n1. Checking model file: {model_path}")
if os.path.exists(model_path):
    print(f"   [OK] Model file exists")
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   [OK] File size: {file_size:.2f} MB")
else:
    print(f"   [FAIL] Model file NOT found!")
    exit(1)

# Test 2: Load TensorFlow/Keras
print(f"\n2. Loading TensorFlow/Keras...")
try:
    import tensorflow as tf
    import keras
    print(f"   [OK] TensorFlow version: {tf.__version__}")
    print(f"   [OK] Keras version: {keras.__version__}")
except Exception as e:
    print(f"   [FAIL] Failed to import TensorFlow: {e}")
    exit(1)

# Test 3: Load the model
print(f"\n3. Loading the model...")
try:
    model = keras.models.load_model(model_path)
    print(f"   [OK] Model loaded successfully!")
except Exception as e:
    print(f"   [FAIL] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Check model architecture
print(f"\n4. Model architecture:")
try:
    print(f"   Model type: {type(model)}")
    if hasattr(model, 'summary'):
        model.summary()
    if hasattr(model, 'input_shape'):
        print(f"   Input shape: {model.input_shape}")
    if hasattr(model, 'output_shape'):
        print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"   Could not display architecture: {e}")

# Test 5: Test inference with dummy data (Colab-aligned)
print(f"\n5. Testing inference with dummy data (Colab-aligned)...")
try:
    # Prefer two-input: frames (64x64x3) and mfcc (40x150)
    inputs = getattr(model, 'inputs', [])
    if len(inputs) >= 2:
        X_frames = np.random.rand(2, 15, 64, 64, 3).astype(np.float32)
        X_mfcc = np.random.rand(2, 40, 150).astype(np.float32)
        print(f"   Frames shape: {X_frames.shape}; MFCC shape: {X_mfcc.shape}")
        predictions = model.predict([X_frames, X_mfcc], verbose=0)
    else:
        X_frames = np.random.rand(2, 15, 64, 64, 3).astype(np.float32)
        print(f"   Frames shape: {X_frames.shape}")
        predictions = model.predict(X_frames, verbose=0)

    print(f"   [OK] Prediction successful!")
    print(f"   Output shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions.flatten()}")
    
    if predictions.shape[-1] == 1 or len(predictions.shape) == 1:
        print(f"   [OK] Output appears to be binary classification")
    else:
        print(f"   [WARN] Expected output shape (N, 1) but got {predictions.shape}")
        
except Exception as e:
    print(f"   [FAIL] Inference failed: {e}")
    print(f"\n   Trying alternative input shape (N, T, features)...")
    try:
        dummy_input_flat = np.random.rand(2, 15, 64*64*3).astype(np.float32)
        print(f"   Alternative input shape: {dummy_input_flat.shape}")
        predictions = model.predict(dummy_input_flat, verbose=0)
        print(f"   [OK] Alternative prediction successful!")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions.flatten()}")
    except Exception as e2:
        print(f"   [FAIL] Alternative inference also failed: {e2}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
