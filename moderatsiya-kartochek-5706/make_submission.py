import torch
import torchvision.transforms as T
import os
from PIL import Image
from tqdm import tqdm
from model6 import Model as M6
from utils import init_model

MODEL_WEIGHTS = "cp.pt"
TEST_IMAGES_DIR = "./data/test"
SUBMISSION_PATH = "./data/submission.csv"

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(M6,MODEL_WEIGHTS,device)
    model.eval()

    img_size = 384
    transform  = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    all_image_names = os.listdir(TEST_IMAGES_DIR)
    all_preds = []
    for image_name in tqdm(all_image_names):
        img_path = os.path.join(TEST_IMAGES_DIR, image_name)
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.sigmoid(output).item() >= 0.5
            all_preds.append(int(pred))

    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name}\t{cl_id}\n")
