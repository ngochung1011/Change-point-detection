import torch
from transformers import MoiraiModel, MoiraiConfig

def finetune_moirai(data, epochs=5, lr=1e-4):
    """Fine-tune MOIRAI model on VN30 dataset."""
    config = MoiraiConfig()
    model = MoiraiModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "moirai_finetuned.pth")
    print("Model fine-tuning completed.")

if __name__ == "__main__":
    train_data = torch.tensor(np.load("processed_vn30_data.npy"))
    finetune_moirai(train_data)