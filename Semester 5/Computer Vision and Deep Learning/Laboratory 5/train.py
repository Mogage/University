import torch
import tqdm


def train_model(model, criterion, optimizer, scheduler, data_loader, device, experiment, train_size, epochs=5):
    model.to(device)
    model.train()

    epochs = tqdm.tqdm(range(epochs), desc="Epochs")

    for epoch in epochs:
        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            experiment.log({'train loss': loss.item()})

        scheduler.step(running_loss / train_size)
        experiment.log({'epoch loss': running_loss / train_size})
        print(f'Epoch {epoch + 1} loss {running_loss / len(data_loader.dataset):.4f}')

    return model
