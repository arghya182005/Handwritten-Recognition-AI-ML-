# Trained model goes here

Place your trained inference model at:

```
model/saved_model/handwriting_crnn.h5
```

This repo does **not** ship a pretrained `.h5` file — training a real
handwriting recognizer needs a labeled dataset (e.g. the IAM
Handwriting Database) and GPU time, neither of which can be baked
into a generated project. `model/train.py` gives you the full
CTC-training scaffold to produce one.

Until a file exists here, the app automatically runs in **demo
mode**: uploads are preprocessed exactly as they would be for a real
model, but the "recognized" text is a simulated result so you can
exercise the full UI (upload, camera, confidence score, history)
immediately. The UI clearly flags demo-mode predictions.
