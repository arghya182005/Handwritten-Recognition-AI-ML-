import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# আপনার মডেল এবং ভোকাব ফাইলের সঠিক পাথ
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.keras')
VOCAB_PATH = os.path.join(BASE_DIR, 'model', 'vocab.json')

# আপনার মডেলের ইনপুট শেপ
IMG_HEIGHT = 64
IMG_WIDTH = 1024