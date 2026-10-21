"""
Train ResNet-18 on CIFAR-10 with a single GPU.

Sample PyTorch script for "Running PyTorch Training Jobs on Explorer" in Fall 2026.
The same script runs in an interactive or batch job and a jupyter session.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_args():
    p = argparse.ArgumentParser(description="ResNet-18 on CIFAR-10")
    p.add_argument("--data-dir", default="./data",
                   help="where CIFAR-10 is stored")
    p.add_argument("--output-dir", default="./output",
                   help="where the checkpoint is saved")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=max(1, available_cpus() - 1),
                   help="DataLoader worker processes (default: CPU cores - 1)")
    return p.parse_args()


def check_gpu():
    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: PyTorch cannot find a GPU\n"
            f"  - PyTorch CUDA build: {torch.version.cuda} "
            "Compare it with 'CUDA Version' in nvidia-smi"
        )
    print(f"PyTorch {torch.__version__} (CUDA build {torch.version.cuda})", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    return torch.device("cuda")


def get_loaders(data_dir, batch_size, num_workers):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    try:
        train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=False, transform=train_tf)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=False, transform=test_tf)
    except RuntimeError:
        raise SystemExit(
            f"ERROR: CIFAR-10 dataset not found in {data_dir}\n"
            "  Please download it first:\n"
            f"  python download_data.py --data-dir {data_dir}"
        )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = torch.zeros((), device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
    return total_loss.item() / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum()
    return correct.item() / len(loader.dataset)


def save_checkpoint(path, model, optimizer, scheduler, epoch):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    device = check_gpu()
    print(f"CPU cores: {available_cpus()} | DataLoader workers: {args.num_workers}", flush=True)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = torchvision.models.resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {ckpt_path}, starting at epoch {start_epoch + 1}", flush=True)

    run_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.perf_counter() - start
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch)

        throughput = len(train_loader.dataset) / train_time
        print(f"epoch {epoch + 1}/{args.epochs} | train loss {train_loss:.4f} | "
              f"test acc {test_acc * 100:.2f}% | train time {train_time:.1f} s | "
              f"{throughput:.0f} img/s", flush=True)

    print(f"Done. Total time {time.perf_counter() - run_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
