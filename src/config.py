# config.py
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")

IMG_SIZE = 320           # small enough for fast GPU inference on GTX1650
NUM_CLASSES = 3          # default; change for dataset (e.g., OxfordPet => 3: background, pet, boundary)
BATCH_SIZE = 8           # adjust if OOM
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# model choices: 'smp_unet' (segmentation_models_pytorch) or 'torchvision_deeplab'
MODEL_BACKEND = "torchvision_deeplab"

# smp options (used only if MODEL_BACKEND=='smp_unet')
SMP_ENCODER = "mobilenet_v2"
SMP_ENCODER_WEIGHTS = "imagenet"
SMP_MODEL = "unet"   # "unet" or "deeplabv3"

# torchvision deeplab options (used if MODEL_BACKEND=='torchvision_deeplab')
# we will use pretrained weights available via torchvision
TV_BACKBONE = "mobilenet_v3_large"

# training & inference
NUM_WORKERS = 4
SAVE_EVERY = 1