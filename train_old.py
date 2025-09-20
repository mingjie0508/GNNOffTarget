import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.dataset import CRISPRoffTDataset
from src.model import CRISPROffTClassifier


# datasets
TRAIN_DATA_ROOT = 'data'
TRAIN_DATA_PATH = 'data/allframe_update_addEpige.txt'
N_HEAD = 4
FEED_FORWARD_DIM = 320

# model
HEADS = 4
EMBED_DIM = 768
HIDDEN_DIM = 16
EMBED_MODEL = 'zhihan1996/DNABERT-S'

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
DROPOUT = 0.0
LR = 1e-4
EPOCHS = 50
WEIGHT_DECAY = 0.0
CHECKPOINT_PATH = f"checkpoints/gatlinkpredictor_gnnofft.pth"


# Dataloader
dataset_train = CRISPRoffTDataset(root=TRAIN_DATA_ROOT)
dataloader_train = DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True
)


# Training Loops
def train(model, dataloader, optimizer, criterion, epochs, checkpoint_path, verbose=True):
    losses_train = []
    for epoch in range(epochs):
        if verbose:
            print('-'*10)
            print(f'Epoch {epoch + 1}/{epochs}:')
        model.train()
        running_loss = 0.0
        for graph in dataloader:
            x = [seq for xx in graph.x for seq in xx] #.to(DEVICE).float()
            edge_index = graph.edge_index.to(DEVICE)
            edge_label_index = graph.edge_label_index.to(DEVICE)
            labels = graph.y.to(DEVICE).float()
            model.zero_grad()
            logits = model(x, edge_index, edge_label_index)
            loss = criterion(logits, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        
        epoch_loss = running_loss / len(dataloader)
        losses_train.append(epoch_loss)
        if verbose:
            print(f"Training loss: {epoch_loss:.2f}")
    # save model checkpoint
    torch.save(
        {'epoch': epoch,
         'model': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'loss': epoch_loss},
        checkpoint_path
    )
    return losses_train


# Training
# model
classifier = CRISPROffTClassifier(
    embed_model=EMBED_MODEL,
    in_channels=EMBED_DIM,
    hidden_channels=HIDDEN_DIM,
    out_channels=HIDDEN_DIM,
    heads=HEADS,
    dropout=DROPOUT,
    local_files_only=True
)
classifier = classifier.to(DEVICE)

# model path
# checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
# classifier.load_state_dict(checkpoint['model'], strict=False)

# optimizer
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# optimizer.load_state_dict(checkpoint['optimizer'])

# loss function
criterion = nn.BCEWithLogitsLoss()

# train model
torch.random.seed()
losses_train = train(
    model=classifier,
    dataloader=dataloader_train,
    optimizer=optimizer,
    criterion=criterion,
    epochs=EPOCHS,
    checkpoint_path=CHECKPOINT_PATH,
    verbose=True
)
