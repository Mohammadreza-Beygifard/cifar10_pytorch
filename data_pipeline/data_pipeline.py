from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def transform_provider(transform_type: str) -> torchvision.transforms:
    if transform_type == "primary":
        return transforms.Compose(
            [
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif transform_type == "test":
        return transforms.Compose(
            [
                transform_provider("primary").transforms[0],
                transforms.ToTensor(),
            ]
        )
    elif transform_type == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomResizedCrop(
                    32, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True
                ),
                transforms.GaussianBlur(kernel_size=3),
            ]
        )

    else:
        raise ValueError(f"Invalid transform type: {transform_type}")


def get_cifar10():
    original_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_provider("test")
    )

    return original_dataset, test_dataset


def get_train_data():
    original_dataset, _ = get_cifar10()

    augmented_data = []
    for original_image, original_label in tqdm(
        original_dataset, desc="Augmenting data"
    ):
        augmented_data.append(
            (transform_provider("primary")(original_image), original_label)
        )
        augmented_image = transform_provider("train")(original_image)
        augmented_data.append(
            (transform_provider("primary")(augmented_image), original_label)
        )

    train_dataset = CustomDataset(augmented_data)
    return train_dataset
