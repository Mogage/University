import matplotlib.pyplot as plt
from evaluate import evaluate_model


def train_model(model,
                criterion,
                optimizer,
                train_data_loader,
                val_data_loader,
                device,
                train_size,
                epochs=5):
    model.to(device)
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_losses.append(running_loss / train_size)
        print(f'Epoch {epoch + 1} train loss {running_loss / train_size:.4f}')

        val_accuracy = evaluate_model(model, val_data_loader, device)
        val_accuracies.append(val_accuracy)

        if (epoch + 1) % 10 == 0:
            model.save(f'model_{val_accuracy * 100:.2f}.pt')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.show()

    return model
