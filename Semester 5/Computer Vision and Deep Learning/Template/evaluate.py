def accuracy_method():
    pass


def evaluate_model(model, data_loader, device):
    model.to(device)
    model.eval()

    accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy()
            labels = labels.cpu().numpy()

            predicted_images = np.argmax(outputs, axis=1)
            ground_truth_images = np.argmax(labels, axis=1)

            accuracy += accuracy_method()

    accuracy /= len(data_loader)
