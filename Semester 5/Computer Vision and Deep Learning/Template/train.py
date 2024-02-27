def train_model(model,
                criterion,
                optimizer,
                train_data_loader,
                val_data_loader,
                device,
                epochs=5):
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()

        for inputs, labels in train_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        evaluate_model(model, val_data_loader, device)

    return model
