import torch
import torch.nn.functional as F

def _masked_accuracy(logits, labels, mask):
    if mask is None or mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    preds = logits[mask].argmax(dim=-1)
    labels = labels[mask]
    correct = (preds == labels).float().sum()
    return correct / labels.numel()

def train_epoch(model, optimizer, data, device):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    mask = data.train_mask & (data.y >= 0)
    loss = F.cross_entropy(logits[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    train_acc = _masked_accuracy(logits, data.y, mask)
    return loss.item(), train_acc.item(), optimizer

def evaluate_network(model, data, device, split: str):
    model.eval()
    mask = getattr(data, f"{split}_mask")
    mask = mask & (data.y >= 0)
    if mask.sum() == 0:
        return 0.0, 0.0
    with torch.no_grad():
        logits = model(data)
        loss = F.cross_entropy(logits[mask], data.y[mask])
        acc = _masked_accuracy(logits, data.y, mask)
    return loss.item(), acc.item()
