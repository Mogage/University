import torch
import tqdm
from evaluate import evaluate_model
from checkpoints import ModelCheckpoint


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                train_data_loader,
                val_data_loader,
                device,
                experiment,
                train_size,
                epochs=5):
    model.to(device)
    model.train()

    epochs = tqdm.tqdm(range(epochs), desc="Epochs")
    checkpoints_callback = ModelCheckpoint()

    for epoch in epochs:
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

            experiment.log({'train loss': loss.item()})

        scores = evaluate_model(model, val_data_loader, device, experiment)

        scheduler.step(scores[1])
        experiment.log({
            'epoch loss': running_loss / train_size,
            'best_miou': scores[1]
        })
        checkpoints_callback(model, epoch, scores[1])
        print(f'Epoch {epoch + 1} loss {running_loss / train_size:.4f}')

    return model
