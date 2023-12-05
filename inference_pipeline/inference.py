from torchvision import transforms
from PIL import Image


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    image = Image.open(image_path)
    input_image = transform(image).unsqueeze(0)
    return input_image
