import torch


def check_performance(model, test_dataset):
    model.to(torch.device("cpu"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataset:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total} %"
    )
