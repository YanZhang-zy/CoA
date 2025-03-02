import os
import glob
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from model import Teacher, Student, Student_x
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

# MODEL_PATH = './model/Teacher_model/Teacher.pth'
# OUTPUT_FOLDER = './outputs/Teacher'

# MODEL_PATH = './model/Student_model/Student.pth'
# OUTPUT_FOLDER = './outputs/Student'

MODEL_PATH = './model/EMA_model/EMA_r.pth'
OUTPUT_FOLDER = './outputs/EMA'


def dehaze(model, image_path, folder):
    haze = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    h, w = haze.shape[2], haze.shape[3]
    haze = Resize((h // 16 * 16, w // 16 * 16), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze)
    out = model(haze)[0].squeeze(0)
    out = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)
    torchvision.utils.save_image(out, os.path.join(folder, os.path.basename(image_path)))


if __name__ == '__main__':

    transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = Teacher().to(device)
    # model = Student().to(device)
    model = Student_x().to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    INPUT_FOLDER = './test'

    images = glob.glob(os.path.join(INPUT_FOLDER, '*jpg')) + glob.glob(os.path.join(INPUT_FOLDER, '*png')) + glob.glob(os.path.join(INPUT_FOLDER, '*jpeg'))

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Rate: {rate_fmt} items/sec"
    with torch.no_grad():
        for image in tqdm(images, bar_format=bar_format, desc="Models are struggling to get out of the fog 😊 :"):
            dehaze(model, image, OUTPUT_FOLDER)
