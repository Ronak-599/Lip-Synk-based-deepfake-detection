import os
import uuid
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, abort

# Optional heavy libs are imported lazily in worker thread to speed app start
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None

ALLOWED_EXTENSIONS = {"mp4", "avi", "flv", "mov", "mkv"}
CHUNK_SIZE_FRAMES = 15
# Decision threshold aligned with Colab notebook
THRESHOLD = 0.52

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-key"),
    MAX_CONTENT_LENGTH=512 * 1024 * 1024,  # 512MB
    UPLOAD_FOLDER=os.path.join("static", "uploads"),
    FRAMES_FOLDER=os.path.join("static", "frames"),
    RESULTS_FOLDER=os.path.join("static", "results"),
    MODELS_FOLDER=os.path.join("models"),
)

# In-memory job store. For production, use Redis or DB.
JOBS: Dict[str, Dict[str, Any]] = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path: str):
    """Load model preferring standalone Keras; fallback to tf.keras.

    This avoids incompatibilities between TensorFlow 2.17.x and Keras 3 by
    using the standalone keras package when available.
    """
    try:
        import keras as _keras  # Standalone Keras (v3)
        model = _keras.models.load_model(model_path)
        return model
    except Exception as _e1:
        # Fallback to tf.keras if standalone keras isn't available/compatible
        try:
            import tensorflow as tf  # noqa: F401  # ensures TF is loaded
            from tensorflow import keras as _keras
            model = _keras.models.load_model(model_path)
            return model
        except Exception as _e2:
            # Re-raise original error with context
            raise RuntimeError(f"Failed to load model via keras and tf.keras: {_e1} | {_e2}")


def background_analyze(job_id: str, video_path: str, app_config: dict):
    """Background worker that runs the full analysis pipeline.

    This implementation aligns preprocessing with the Colab script:
    - 64x64 color lip crops using a simple lower-center mouth crop
    - chunks of 15 frames
    - audio MFCCs with 40 coefficients and 150 time steps, repeated per chunk
    - final decision uses average confidence vs THRESHOLD (0.52)
    """
    # Heavy imports here to avoid slowing initial app load
    import numpy as np
    import cv2
    # MediaPipe is optional; fall back gracefully if unavailable
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        # Match Colab config: no refine_landmarks
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        has_mp = True
        print("[DEBUG] MediaPipe loaded successfully")
    except Exception as mp_exc:
        mp = None  # type: ignore
        face_mesh = None
        has_mp = False
        print(f"[WARN] MediaPipe unavailable, will use simple fallback cropping: {mp_exc}")

    from moviepy.editor import VideoFileClip
    import librosa
    import soundfile as sf  # noqa: F401 (ensures libsndfile backend is present)

    model_path = os.path.join(app_config["MODELS_FOLDER"], "cremad_model_finetuned_5600_5953.keras")

    def to_web_path(p: str | None):
        if not p:
            return None
        p_norm = p.replace("\\", "/")
        idx = p_norm.lower().find("/static/")
        if idx != -1:
            return p_norm[idx + 1:]  # drop leading slash to make 'static/...'
        return p_norm

    start_time = time.time()
    job = JOBS[job_id]
    job.update(
        {
            "status": "running",
            "progress": 5,
            "message": "Loading model...",
            "started_at": datetime.utcnow().isoformat() + "Z",
        }
    )

    try:
        # 1) Load model
        try:
            print(f"[DEBUG] Loading model from: {model_path}")
            model = load_model(model_path)
            model_loaded = True
            # Log model signature for debugging
            try:
                in_shapes = [getattr(inp, "shape", None) for inp in getattr(model, "inputs", [])]
                out_shapes = [getattr(out, "shape", None) for out in getattr(model, "outputs", [])]
                print(f"[DEBUG] Model loaded successfully! inputs={in_shapes}, outputs={out_shapes}")
            except Exception:
                print("[DEBUG] Model loaded successfully! (shapes unavailable)")
            job.update({"progress": 10, "message": "Extracting audio..."})
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")

            class DummyModel:
                def predict(self, x, verbose=0):  # noqa: ANN001
                    import numpy as _np

                    n = x.shape[0] if hasattr(x, "shape") else len(x)
                    return _np.full((n, 1), 0.5, dtype=_np.float32)

            model = DummyModel()
            model_loaded = False
            job.update(
                {
                    "progress": 10,
                    "message": "Model not found/failed to load. Using dummy predictions.",
                }
            )

        # 2) Load video and extract audio
        print(f"[DEBUG] Loading video: {video_path}")
        clip = VideoFileClip(video_path)
        fps = clip.fps or 30
        duration = clip.duration or 0.0
        print(f"[DEBUG] Video loaded: {duration:.2f}s, {fps} fps")

        audio_tmp_path = os.path.join(app_config["RESULTS_FOLDER"], f"{job_id}_audio.wav")
        if clip.audio is not None:
            print("[DEBUG] Extracting audio...")
            clip.audio.write_audiofile(audio_tmp_path, fps=16000, verbose=False, logger=None)
            y, sr = librosa.load(audio_tmp_path, sr=16000, mono=True)
            print(f"[DEBUG] Audio extracted: {len(y)} samples at {sr} Hz")
        else:
            print("[DEBUG] No audio track found")
            y, sr = None, None

        job.update({"progress": 20, "message": "Extracting frames and lips..."})

        # 3) Extract frames and build 15-frame chunks
        print("[DEBUG] Starting frame extraction...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        read_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30
        chunk_size = CHUNK_SIZE_FRAMES
        print(f"[DEBUG] Total frames: {total_frames}, FPS: {read_fps}")

        frame_idx = 0
        chunk_idx = 0
        chunks: list[np.ndarray] = []
        keyframe_paths: list[str] = []

        # Colab-style simple mouth crop around lower-center of the frame
        def simple_mouth_crop(frame_bgr: np.ndarray) -> np.ndarray | None:
            h, w, _ = frame_bgr.shape
            cx, cy = w // 2, int(h * 0.65)
            size = int(h * 0.25)
            y1, y2 = max(cy - size, 0), min(cy + size, h)
            x1, x2 = max(cx - size, 0), min(cx + size, w)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                return None
            return crop

        frames_current_chunk: list[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Gate by face detection like in Colab; then do simple mouth crop
            lip_roi = None
            try:
                if has_mp and face_mesh is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = face_mesh.process(rgb)  # type: ignore[attr-defined]
                    if res.multi_face_landmarks:
                        lip_roi = simple_mouth_crop(frame)
            except Exception:
                lip_roi = None

            # If MediaPipe unavailable, still attempt simple crop
            if lip_roi is None and not has_mp:
                lip_roi = simple_mouth_crop(frame)

            if lip_roi is None:
                continue  # skip frame when we can't crop a valid mouth ROI

            lip_roi = cv2.resize(lip_roi, (64, 64))  # keep color (BGR) as in Colab
            frames_current_chunk.append(lip_roi)

            # Save a keyframe for this frame if first of chunk
            if len(frames_current_chunk) == 1:
                key_path = os.path.join(app_config["FRAMES_FOLDER"], f"{job_id}_chunk{chunk_idx:04d}.jpg")
                try:
                    cv2.imwrite(key_path, lip_roi)
                    keyframe_paths.append(key_path)
                except Exception:
                    keyframe_paths.append("")

            if len(frames_current_chunk) == chunk_size:
                chunks.append(np.stack(frames_current_chunk, axis=0))  # (15, 64, 64, 3)
                frames_current_chunk = []
                chunk_idx += 1

            if total_frames:
                job["progress"] = 20 + int(30 * frame_idx / total_frames)
                job["message"] = f"Processing frames {frame_idx}/{total_frames}"

        # Drop remainder frames (only keep full 15-frame chunks) to match Colab
        if frames_current_chunk:
            print(
                f"[DEBUG] Dropping remainder frames: {len(frames_current_chunk)} (not a full chunk)"
            )

        print(f"[DEBUG] Frame extraction complete. Total chunks: {len(chunks)}")
        cap.release()
        if has_mp and face_mesh is not None:
            try:
                face_mesh.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        clip.close()

        job.update({"progress": 55, "message": "Preparing features..."})

        # 4) Build model inputs per Colab: frames as color 64x64, no scaling
        if len(chunks) == 0:
            raise RuntimeError("No valid lip chunks could be extracted from the video.")
        X_frames = np.stack(chunks, axis=0).astype(np.float32)  # (N, 15, 64, 64, 3)

        # Audio features per chunk (rough alignment by duration)
        mfcc_chunks = None
        waveform = None
        if y is not None and sr is not None and duration > 0:
            # Colab-style: compute full MFCC with 40 coeffs and take first 150 time steps for all chunks
            mfcc_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            if mfcc_full.shape[1] < 150:
                mfcc_full = np.pad(
                    mfcc_full, ((0, 0), (0, 150 - mfcc_full.shape[1])), mode="constant"
                )
            else:
                mfcc_full = mfcc_full[:, :150]
            mfcc_chunks = (
                np.repeat(mfcc_full[np.newaxis, :, :], X_frames.shape[0], axis=0).astype(np.float32)
            )  # (N, 40, 150)

            # Waveform visualization (optional)
            try:
                frame_len = max(1, int(sr * 0.02))  # 20ms
                hop_len = frame_len
                rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len).squeeze()
                times = librosa.times_like(rms, sr=sr, hop_length=hop_len)
                max_r = float(np.max(rms)) if rms.size else 1.0
                if max_r <= 0:
                    max_r = 1.0
                y_plot = (rms / max_r).astype(np.float32)
                # Downsample to <= 1000 points
                max_pts = 1000
                if len(times) > max_pts:
                    step = int(np.ceil(len(times) / max_pts))
                    times = times[::step]
                    y_plot = y_plot[::step]
                waveform = {
                    "x": [round(float(t), 3) for t in times.tolist()],
                    "y": [round(float(v), 3) for v in y_plot.tolist()],
                }
            except Exception:
                waveform = None

        job.update({"progress": 70, "message": "Running inference..."})

        # 5) Infer. Predict like in Colab: multimodal [frames, mfcc]
        print(f"[DEBUG] Running model inference on {len(X_frames)} chunks...")
        print(f"[DEBUG] Frames input shape: {X_frames.shape}")
        if mfcc_chunks is not None:
            print(f"[DEBUG] MFCC input shape: {mfcc_chunks.shape}")

        try:
            if mfcc_chunks is not None:
                preds = model.predict([X_frames, mfcc_chunks], verbose=0)
            else:
                preds = model.predict(X_frames, verbose=0)
        except Exception as e:
            print(f"[ERROR] Prediction failed with error: {e}")
            preds = None

        if preds is None:
            # Fallback dummy scores to avoid crash
            print("[WARNING] Using fallback dummy predictions")
            preds = np.full((len(X_frames), 1), 0.5, dtype=np.float32)

        preds = np.array(preds)
        # Normalize outputs to probability per chunk in [0,1]
        # Common cases: (N,1), (N,), (N,2), logits or probabilities
        if preds.ndim == 2 and preds.shape[1] == 1:
            probs = preds.squeeze(1)
        elif preds.ndim == 2 and preds.shape[1] == 2:
            # Softmax if not already
            row_sums = np.sum(preds, axis=1, keepdims=True)
            # If sums are not approx 1 or contain negatives, apply softmax
            if not np.allclose(row_sums, 1.0, atol=1e-3) or np.any(preds < 0):
                e = np.exp(preds - np.max(preds, axis=1, keepdims=True))
                soft = e / np.sum(e, axis=1, keepdims=True)
                probs = soft[:, 1]
            else:
                probs = preds[:, 1]
        elif preds.ndim == 1:
            # Possibly logits; squash to [0,1] if outside range
            if np.any((preds < 0) | (preds > 1)):
                probs = 1.0 / (1.0 + np.exp(-preds))
            else:
                probs = preds
        elif preds.ndim == 0:
            probs = np.array([float(preds)])
        else:
            probs = preds.squeeze()

        if isinstance(probs, np.ndarray) and probs.ndim == 0:
            probs = np.array([float(probs)])

        print(
            f"[DEBUG] Final predictions shape: {np.shape(probs)}, values: {probs[:5] if np.size(probs) > 5 else probs}"
        )

        job.update({"progress": 85, "message": "Aggregating results..."})

        results: list[dict[str, Any]] = []
        for i, p in enumerate(probs):
            p = float(p)
            # Match Colab mapping: FAKE if prob <= THRESHOLD else REAL
            label = "FAKE" if p <= THRESHOLD else "REAL"
            start_s = i * (chunk_size / read_fps)
            end_s = (i + 1) * (chunk_size / read_fps)
            keyframe = keyframe_paths[i] if i < len(keyframe_paths) else None
            results.append(
                {
                    "chunk": i,
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "confidence": round(p, 4),
                    "label": label,
                    "keyframe": to_web_path(keyframe) if keyframe else None,
                }
            )

        avg_conf = float(np.mean(probs)) if len(probs) else 0.5
        # Final decision per Colab: average then compare to THRESHOLD
        final_label = "FAKE" if avg_conf <= THRESHOLD else "REAL"

        processing_time = round(time.time() - start_time, 2)

        job_data = {
            "job_id": job_id,
            "video_path": to_web_path(video_path),
            "audio_path": to_web_path(audio_tmp_path) if clip.audio is not None else None,
            "fps": read_fps,
            "duration": duration,
            "total_frames": total_frames,
            "chunk_size_frames": chunk_size,
            "predictions": results,
            "avg_confidence": round(avg_conf, 4),
            "final_label": final_label,
            "processing_time": processing_time,
            "waveform": waveform,
            "model_loaded": model_loaded,
        }

        # Save JSON
        json_path = os.path.join(app_config["RESULTS_FOLDER"], f"{job_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2)

        print(f"[DEBUG] Results saved to: {json_path}")
        print(f"[DEBUG] Final verdict: {final_label}, Avg confidence: {avg_conf:.4f}")

        job.update(
            {
                "status": "completed",
                "progress": 100,
                "message": "Analysis complete",
                "result_path": json_path,
            }
        )

    except Exception as e:
        print(f"[ERROR] Processing failed with exception: {e}")
        import traceback

        traceback.print_exc()
        job.update(
            {
                "status": "error",
                "progress": 100,
                "message": f"Error: {e}",
            }
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"]) 
def analyze():
    if "video" not in request.files:
        return redirect(url_for("index"))
    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["FRAMES_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)

    ext = file.filename.rsplit(".", 1)[1].lower()
    job_id = uuid.uuid4().hex
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}.{ext}")
    file.save(save_path)

    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    t = threading.Thread(target=background_analyze, args=(job_id, save_path, app.config), daemon=True)
    t.start()

    return redirect(url_for("result_page", job_id=job_id))


@app.route("/status/<job_id>")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify({
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
    })


@app.route("/result/<job_id>")
def result_page(job_id: str):
    return render_template("result.html", job_id=job_id)


@app.route("/api/result/<job_id>")
def api_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job.get("status") != "completed":
        return jsonify({"status": job.get("status"), "message": job.get("message")}), 202
    json_path = job.get("result_path")
    if not json_path or not os.path.exists(json_path):
        return jsonify({"error": "result not found"}), 404
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/download/csv/<job_id>")
def download_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        abort(404)
    json_path = job.get("result_path")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    import csv
    csv_path = os.path.join(app.config["RESULTS_FOLDER"], f"{job_id}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["chunk", "start", "end", "confidence", "label", "keyframe"])
        for r in data.get("predictions", []):
            writer.writerow([r["chunk"], r["start"], r["end"], r["confidence"], r["label"], r.get("keyframe", "")])
    return send_file(csv_path, as_attachment=True, download_name=f"{job_id}.csv")


if __name__ == "__main__":
    # Ensure folders exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["FRAMES_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODELS_FOLDER"], exist_ok=True)

    # App run
    app.run(host="0.0.0.0", port=5000, debug=True)