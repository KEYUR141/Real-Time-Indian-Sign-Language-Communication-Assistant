# ISL presentation frontend

A focused React demo connected to the existing MediaPipe + temporal Transformer pipeline.

## One-time installation

Run these yourself because dependency installation can take time:

```powershell
uv sync --active
cd frontend
npm install
cd ..
```

## Start the presentation

Terminal 1 — API and trained model:

```powershell
uv run --active python scripts/run_api.py `
  --checkpoint "E:\Hugging_Face\INCLUDE\processed\mediapipe_v1\models\full_include\transformer_final_seed_42\selected_final.pt" `
  --device cuda
```

Terminal 2 — React:

```powershell
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite development server proxies `/api` to
`http://127.0.0.1:8000`.

## Suggested five-minute demo

1. Point out the online/CUDA/262-class status strip.
2. Briefly explain the five-step pipeline at the bottom.
3. Upload a known test video for the most reliable live prediction.
4. Show the predicted sign, confidence, landmark coverage, and latency.
5. Close with the held-out test metrics: 94.36% accuracy and 94.47% Macro F1.

The **Show live points** control runs MediaPipe Hands and Pose in the browser and draws connected
landmarks over the mirrored camera preview. Live capture sends those detected points directly to
the Python API, where the same sampling, normalization, rejection, and trained Transformer used by
the desktop webcam implementation make the final decision. This avoids browser-video codec errors.

The browser prefers MP4 recording when supported and falls back to WEBM. Browser/OpenCV codec
support varies, so keep the tested MOV clip ready for the presentation.

For a reliable live attempt, stand in even lighting with the upper torso and both hands visible.
Wait for `LIVE POINTS`, press **Capture sign**, prepare during the countdown, and perform exactly
one complete sign after recording begins. Test a small set of visually distinctive signs before
the presentation and keep one known test clip as a transparent fallback.
