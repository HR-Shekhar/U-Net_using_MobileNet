import torch
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T

def get_voc_dataloader(batch_size=4):
    transform_img = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    transform_mask = T.Compose([
        T.Resize((256, 256)),
        T.PILToTensor(),
    ])

    train = VOCSegmentation(
        root="./data", year="2012", image_set="train", download=True,
        transform=transform_img, target_transform=transform_mask
    )
    val = VOCSegmentation(
        root="./data", year="2012", image_set="val", download=True,
        transform=transform_img, target_transform=transform_mask
    )

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader
