import torch
from torchvision import transforms
from PIL import Image


def run_inference(model, image_path):
    return _get_cifar10_class(_inference(model, _preprocess_image(image_path)))


def _preprocess_image(image_path):
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


def _get_cifar10_class(class_index):
    if class_index == 0:
        return "airplane"
    elif class_index == 1:
        return "automobile"
    elif class_index == 2:
        return "bird"
    elif class_index == 3:
        return "cat"
    elif class_index == 4:
        return "deer"
    elif class_index == 5:
        return "dog"
    elif class_index == 6:
        return "frog"
    elif class_index == 7:
        return "horse"
    elif class_index == 8:
        return "ship"
    elif class_index == 9:
        return "truck"
    else:
        return "Unknown class"


def _inference(model, input_image):
    model.eval()
    with torch.no_grad():
        output = model(input_image)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()
