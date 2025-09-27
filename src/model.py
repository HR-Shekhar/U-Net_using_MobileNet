# model.py
import torch
from config import MODEL_BACKEND, SMP_ENCODER, SMP_ENCODER_WEIGHTS, SMP_MODEL, NUM_CLASSES
import segmentation_models_pytorch as smp

def get_smp_model():
    if SMP_MODEL.lower() == "unet":
        return smp.Unet(
            encoder_name=SMP_ENCODER,
            encoder_weights=SMP_ENCODER_WEIGHTS,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,  # logits
        )
    else:
        return smp.DeepLabV3(
            encoder_name=SMP_ENCODER,
            encoder_weights=SMP_ENCODER_WEIGHTS,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        )

def get_torchvision_deeplab(num_classes=NUM_CLASSES, pretrained=True):
    # Import here to keep optional dependency flexible
    import torchvision
    weights = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights, progress=True)
    # Adjust classifier for different number of classes (weights were for 21 classes on Pascal subset)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=1)
    return model

def get_model():
    if MODEL_BACKEND == "smp_unet":
        return get_smp_model()
    elif MODEL_BACKEND == "torchvision_deeplab":
        return get_torchvision_deeplab()
    else:
        raise ValueError("Unknown MODEL_BACKEND")
