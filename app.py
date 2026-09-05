import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

import config
from utils.preprocessing import preprocess_image
from utils.decoder import ctc_greedy_decode

# =======================================================
# 🛠️ COMPATIBILITY PATCH (kept from original app.py)
# Some saved-model configs include a 'quantization_config' kwarg on
# Dense layers that this TensorFlow/Keras version's Dense.__init__
# does not accept, which raises an error on tf.keras.models.load_model.
#
# CAUTION: if the model was genuinely trained/saved as a quantized
# model, silently dropping this kwarg discards quantization info and
# can change effective weight values on load — which would degrade
# or corrupt predictions independently of everything else in this
# file. If predictions are still wrong after the vocab fix below,
# the real fix is installing the exact TF/Keras version used during
# training instead of relying on this patch.
# =======================================================
original_dense_init = tf.keras.layers.Dense.__init__

def safe_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = safe_dense_init
# =======================================================

app = Flask(__name__)

# -------------------------------------------------------
# 1. Vocabulary loading
# -------------------------------------------------------
# vocab.json stores char_to_num as {character: index}. We rebuild
# `charset` as a list where charset[i] is the character trained for
# class i, so it exactly matches num_to_char in vocab.json and
# therefore exactly matches the label encoding used during training.
#
# *** THIS IS THE FIX FOR THE GARBLED PREDICTIONS ***
# The previous version of this file built this correct charset and
# then immediately overwrote it with a hardcoded, differently
# ordered alphabet ("abcdefg...ABCDEFG...0123456789..."). Since
# vocab.json is NOT alphabetically ordered (space=0, punctuation
# =1-13, digits=14-23, ':'=24, ';'=25, '?'=26, A-Z=27-52, a-z=53-78),
# every predicted class index was being decoded to the wrong
# character. That mismatch — not the model, not the image
# preprocessing — is why "Mr." came out as ";B;". The hardcoded
# overwrite has been removed.
print("Loading vocabulary...")
charset = []
blank_index = None

try:
    with open(config.VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    mapping_dict = vocab_data.get('char_to_num', vocab_data)

    if isinstance(mapping_dict, dict):
        # Sort strictly by numeric index so charset[i] is guaranteed
        # to be the character the model was trained to predict at
        # position i. (vocab.json values are already ints.)
        sorted_chars = sorted(mapping_dict.items(), key=lambda item: int(item[1]))
        charset = [char for char, _ in sorted_chars]
    elif isinstance(vocab_data, list):
        charset = vocab_data
    else:
        raise ValueError("vocab.json is not in a recognized format")

    # The model's final layer outputs one extra class beyond the real
    # characters: the CTC blank. vocab.json declares this separately
    # as "pad_token" (79 = one past the last real character index,
    # 78). Extend charset with an empty string at that index so the
    # blank class decodes to "" instead of a wrong/crashing lookup.
    pad_token_index = vocab_data.get('pad_token')
    if pad_token_index is not None:
        while len(charset) <= pad_token_index:
            charset.append('')
        blank_index = pad_token_index

except Exception as e:
    print("Error loading vocabulary:", str(e))
    raise SystemExit(f"Cannot start app: failed to load {config.VOCAB_PATH}: {e}")

print(f"Vocab loaded! {len(charset)} classes total (blank/pad index: {blank_index}).")

# -------------------------------------------------------
# 2. Model loading
# -------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
print("Model loaded successfully!")

# Sanity check: the model's output class count must match the
# vocabulary size, or decoding will silently produce wrong text no
# matter how correct everything else is.
try:
    model_output_classes = model.output_shape[-1]
    if model_output_classes != len(charset):
        print(
            f"WARNING: model outputs {model_output_classes} classes but "
            f"charset has {len(charset)} entries. These MUST match — "
            f"re-check vocab.json against the training script's "
            f"num_classes / vocabulary size."
        )
except Exception:
    pass  # output_shape isn't always introspectable for every model type


@app.route('/')
def index():
    return render_template('index.html', demo_mode=False)


def handle_prediction():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        file = request.files['image']
        image_bytes = file.read()

        # -------------------------------------------------------
        # 3. Preprocessing — must exactly match the training pipeline
        # -------------------------------------------------------
        # NOTE: preprocess_image() lives in utils/preprocessing.py,
        # which was not available for this review. Everything below
        # assumes it returns an array shaped either
        # (1, IMG_HEIGHT, IMG_WIDTH) or (1, IMG_HEIGHT, IMG_WIDTH, 1),
        # already grayscale and normalized the same way training
        # images were. If predictions are still wrong after this fix,
        # that file is the next place to check — grayscale
        # conversion, aspect-ratio-preserving resize, which side gets
        # padded and with what fill value, and the normalization
        # formula (e.g. /255.0 vs (x/127.5)-1) all need to be
        # pixel-for-pixel identical to what generated your training
        # data.
        model_input, _ = preprocess_image(
            image_bytes,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH
        )

        if len(model_input.shape) == 3:
            model_input = np.expand_dims(model_input, axis=-1)

        # Model expects width as the time/sequence axis: (batch, W, H, 1).
        # preprocess_image is expected to return (batch, H, W, 1), so
        # height and width are transposed here.
        if model_input.shape[1] == config.IMG_HEIGHT and model_input.shape[2] == config.IMG_WIDTH:
            model_input = np.transpose(model_input, (0, 2, 1, 3))

        # Defensive check against the model's actual expected input
        # shape, so a mismatch shows up in the logs instead of just
        # producing silently wrong predictions.
        expected_shape = model.input_shape
        if expected_shape and len(expected_shape) == 4:
            if (expected_shape[1] not in (None, model_input.shape[1]) or
                    expected_shape[2] not in (None, model_input.shape[2])):
                print(
                    f"WARNING: model expects input shape {expected_shape} "
                    f"but got {model_input.shape}."
                )

        # -------------------------------------------------------
        # 4. Prediction
        # -------------------------------------------------------
        preds = model.predict(model_input)
        print("Raw Model Predictions Shape:", preds.shape)

        # -------------------------------------------------------
        # 5. CTC decoding
        # -------------------------------------------------------
        # NOTE: ctc_greedy_decode() lives in utils/decoder.py, also
        # not available for this review. It must treat blank_index
        # (printed above at startup) as the CTC blank — i.e. skip it
        # rather than appending charset[blank_index] to the output,
        # and collapse repeated non-blank characters per standard CTC
        # greedy decoding.
        recognized_text, confidence = ctc_greedy_decode(preds, charset)
        print("Decoded Text:", repr(recognized_text), "Confidence:", confidence)

        return jsonify({
            'success': True,
            'text': recognized_text,
            'confidence': confidence
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500


# Both routes stay active so the frontend works regardless of which
# endpoint it calls.
@app.route('/predict', methods=['POST'])
def predict():
    return handle_prediction()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    return handle_prediction()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
