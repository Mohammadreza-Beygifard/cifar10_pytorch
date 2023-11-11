import torchvision
import torchvision.transforms as transforms

def get_cifar10():
    transform_primary = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    original_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transforms.ToTensor())

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transform_primary
    ])

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    return original_train_dataset, test_dataset