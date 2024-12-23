import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from model import GCN

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


MODELS = ['gcn', 'graphsage', 'gat', 'gin']
model_name = 'gcn'
LOG_STEP = 10
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
NUM_EPOCHS = 500


def get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout) -> torch.nn.Module:
    assert model_name in MODELS, f'not implemented model: {model_name}'

    if model_name == 'gcn':
        return GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout
        )


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()

    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                    transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    model = get_model(
        model_name,
        data.num_features,
        HIDDEN_DIM,
        dataset.num_classes,
        NUM_LAYERS,
        DROPOUT
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 1 + NUM_EPOCHS):
        loss = train(model, data, train_idx, optimizer)
        result = test(model, data, split_idx, evaluator)

        if epoch % LOG_STEP == 0:
            train_acc, valid_acc, test_acc = result
            print(
                f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}%'
            )


if __name__ == '__main__':
    main()