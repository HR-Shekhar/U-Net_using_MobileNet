# infer.py
import cv2, torch, numpy as np
from model import get_model
from config import IMG_SIZE, DEVICE, NUM_CLASSES
import torchvision.transforms as T

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    # ImageNet normalization
    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))
    return torch.tensor(img).unsqueeze(0).float()

def color_map(num_classes):
    # simple color map
    import random
    random.seed(42)
    cmap = [(0,0,0)]
    for i in range(1, num_classes):
        cmap.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    return np.array(cmap, dtype=np.uint8)

def visualize(frame, mask, alpha=0.6):
    cmap = color_map(NUM_CLASSES)
    colored = cmap[mask]
    colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    blended = (frame * (1-alpha) + colored * alpha).astype(np.uint8)
    return blended

def run(weights_path):
    device = DEVICE
    model = get_model()
    ckpt = torch.load(weights_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    cap = cv2.VideoCapture(0)
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            inp = preprocess(frame).to(device)
            out = model(inp)
            logits = out['out'] if isinstance(out, dict) else out
            pred = torch.softmax(logits, dim=1).argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            vis = visualize(frame, pred)
            cv2.imshow("seg", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python infer.py path/to/ckpt.pth")
    else:
        run(sys.argv[1])