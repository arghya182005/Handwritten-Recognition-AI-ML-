# Scriptura — Handwritten Text Recognition

A full-stack web app for recognizing handwritten text from an
uploaded photo or a live camera capture, built on a CNN + CRNN
(convolutional + bidirectional-LSTM, CTC-decoded) TensorFlow model.

## Stack

| Layer     | Tech |
|-----------|------|
| Frontend  | HTML5, CSS3 (custom design system), Bootstrap 5, vanilla JavaScript |
| Backend   | Python, Flask |
| AI model  | TensorFlow / Keras — CNN feature extractor + BiLSTM (CRNN) + CTC decoding, saved as `.h5` |

## Features

- Drag-and-drop **or** click-to-browse image upload
- Live **camera capture** (`getUserMedia`) with retake support
- Server-side image preprocessing (denoise, adaptive threshold, resize/pad)
- OCR prediction with a **confidence score**
- Local **prediction history** (stored in the browser, with thumbnails)
- Fully responsive layout, **dark mode** with persisted preference
- An animated "ink reveal" transcription effect and a live confidence meter

## Project structure

```
handwriting-recognition-app/
├── app.py                     # Flask app + /api/predict endpoint
├── config.py                  # paths, image size, character set
├── requirements.txt
├── model/
│   ├── crnn_model.py           # CNN + BiLSTM architecture (inference model)
│   ├── train.py                # CTC training scaffold
│   └── saved_model/
│       └── handwriting_crnn.h5  # <- put your trained model here (not included)
├── utils/
│   ├── preprocessing.py        # image -> model tensor
│   └── decoder.py               # CTC greedy decode -> text + confidence
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/
└── templates/
    └── index.html
```

## Getting started

```bash
cd handwriting-recognition-app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## About the model — please read

Training a real handwriting-recognition model needs a labeled dataset
(the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
is the usual choice) and real GPU time — that can't be produced as
part of generating this project.

So, out of the box:

- **No `.h5` file is included.** `app.py` looks for one at
  `model/saved_model/handwriting_crnn.h5`.
- If it isn't found, the app automatically starts in **demo mode**:
  every image still goes through the *real* preprocessing pipeline,
  but the returned "recognition" is a simulated result. The UI shows
  a clear banner whenever a response was produced this way.
- `model/crnn_model.py` defines the real architecture, and
  `model/train.py` is a ready-to-fill CTC-training scaffold. Once you
  train a model and save it to the path above, restart the app — it
  will load automatically and demo mode turns off.

## API

`POST /api/predict` — multipart form field `image` → 
```json
{
  "success": true,
  "text": "the quick brown fox",
  "confidence": 0.91,
  "demo_mode": false,
  "preview_image": "data:image/png;base64,...",
  "timestamp": "2026-07-26T12:00:00Z"
}
```

`GET /api/health` → `{"status": "ok", "demo_mode": false}`

## Notes

- Uploads are processed in memory and are **not** written to disk by
  default (`static/uploads/` is left ready if you'd rather persist
  originals — just add a `file.save(...)` call in `app.py`).
- The character set, image dimensions, and model path are all in
  `config.py` if you need to adapt them to a different trained model.
