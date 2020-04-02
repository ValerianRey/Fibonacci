import torch.nn.functional as F
import torch

# Get Min and max of x tensor, and stores it
def update_stats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)
    batch_size = max_val.shape[0]

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": batch_size}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += batch_size


# Reworked Forward Pass to access activation Stats through update_stats function
def gather_activation_stats(model, x, stats):
    update_stats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = model.conv1(x)
    x = F.relu(x)

    update_stats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = model.conv2(x)
    x = F.max_pool2d(x, 2)
    x = model.dropout1(x)
    x = torch.flatten(x, 1)

    update_stats(x, stats, 'fc1')
    x = model.fc1(x)
    x = F.relu(x)
    x = model.dropout2(x)

    update_stats(x, stats, 'fc2')


# Entry function to get stats of all functions.
def gather_stats(model, test_loader):
    device = 'cuda:0'
    print("Gathering stats...")
    model.eval()
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            gather_activation_stats(model, data, stats)

    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = {"max": value["max"] / value["total"], "min": value["min"] / value["total"]}

    print("Gathering completed")
    return final_stats

