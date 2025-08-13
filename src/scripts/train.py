from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """
    Trains the JointBERT model for joint intent classification and slot filling.

    The training loop:
    - Sends the model to the specified device (CPU/GPU).
    - Iterates over multiple epochs.
    - For each batch:
        - Moves inputs and labels to the device.
        - Runs a forward pass through the model to get predictions.
        - Calculates the combined loss using the provided criterion.
        - Performs backpropagation and optimizer step to update model weights.
        - Tracks training loss.
    - Prints the average training loss per epoch.

    Args:
        model (nn.Module): The JointBERT model instance.
        train_loader (DataLoader): DataLoader providing the training data.
        criterion (nn.Module): The loss function (e.g., JointLoss combining intent & slot losses).
        optimizer (torch.optim.Optimizer): The optimizer (e.g., AdamW).
        device (str): "cuda" for GPU or "cpu" for CPU training.
        num_epochs (int, optional): Number of training epochs. Default is 5.

    Returns:
        None
        (Trains the model in-place and prints progress.)
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_label'].to(device)
            slot_labels = batch['slot_labels'].to(device)

            optimizer.zero_grad()
            intent_logits, slot_logits = model(input_ids, attention_mask)
            loss = criterion(intent_logits, slot_logits, intent_labels, slot_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} completed. Avg Training Loss: {avg_loss:.4f}")