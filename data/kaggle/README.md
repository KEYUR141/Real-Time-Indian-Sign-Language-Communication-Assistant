# Kaggle Datasets

This folder is reserved for datasets downloaded from Kaggle.

## ISL Character Dataset

- Dataset handle: `prathumarikeri/indian-sign-language-isl`
- Source: <https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl>
- Format: Static character-level images
- Intended project use: Dataset inspection, MediaPipe experiments, and an optional alphabet-recognition baseline
- Not intended as the primary dataset for dynamic word-level recognition

## Download

Run these commands from the repository root:

```powershell
python -m pip install kaggle
kaggle auth login
kaggle datasets download prathumarikeri/indian-sign-language-isl --path data/kaggle/indian-sign-language-isl --unzip
```

Do not commit downloaded dataset files to Git. Keep this README and the local ignore rules in place.

