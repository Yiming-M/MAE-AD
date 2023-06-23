import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from argparse import ArgumentParser
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from mae import mae_vit_large_patch16
from dataset import MVTecDataset
from losses import RIADLoss, RIADScore


rng = np.random.default_rng()
parser = ArgumentParser(description="Training MAE-based anomaly detection model on the MVTec AD dataset.")
parser.add_argument("--root", type=str, default="./data", help="Path to the MVTec AD dataset root directory.")
parser.add_argument("--category", type=str, default="bottle", help="Category name of the MVTec AD dataset.")
parser.add_argument("--resize", type=tuple, default=(256, 256), help="Image size for training.")
parser.add_argument("--crop_size", type=tuple, default=(224, 224), help="Image size for training.")

parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size for testing.")

parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay.")
parser.add_argument("--num_epochs", type=int, default=500, help="The number of epochs to train.")

parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for dataloader.")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")


def init_seeds(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def main():
    args = parser.parse_args()
    args.num_workers = os.cpu_count() + args.num_workers - 1 if args.num_workers < 0 else args.num_workers
    assert args.num_workers > 0

    args.checkpoint_dir = os.path.join("checkpoints", "mae_pretrained", args.category)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    args.nprocs = torch.cuda.device_count()
    print(f"Number of GPUs: {args.nprocs}")

    for key, value in vars(args).items():
        print(f"Training param setting {key} = {value}")
    
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def setup(local_rank, nprocs):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=local_rank, world_size=nprocs)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main_worker(local_rank, nprocs, args):
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)
    print(f"Rank {local_rank} process initialized.")
    device = torch.device(f"cuda:{local_rank}")

    model = mae_vit_large_patch16(img_size=224).to(device)
    loss_fn = RIADLoss().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1, verbose=True)

    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        curr_epoch = checkpoint["curr_epoch"]
        hist_costs = checkpoint["hist_costs"]
        hist_scores = checkpoint["hist_scores"]
        best_scores = checkpoint["best_scores"]
        print(f"Found checkpoint at epoch {curr_epoch} with best scores {best_scores}. Loading...")
    else:
        curr_epoch = 0
        hist_costs = []
        best_scores = {
            "auc_roc": 0.,
            "auc_pr": 0.,
        }
        hist_scores = {score: [] for score in best_scores.keys()}
        print("No checkpoint found. Starting from scratch...")

    model_ddp = DDP(model, device_ids=[local_rank])
    train_batch_size, num_workers = int(args.train_batch_size // nprocs), int(args.num_workers // nprocs)
    train_dataset = MVTecDataset(
        root=args.root,
        split="train",
        category=args.category,
        resize=args.resize,
        crop_size=args.crop_size,
    )
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if local_rank == 0:
        # score_fn = MultiScaleStructuralSimilarityIndexMeasure(reduction="none").to(device)
        score_fn = RIADScore().to(device)
        test_dataset = MVTecDataset(
            root=args.root,
            split="test",
            category=args.category,
            resize=args.resize,
            crop_size=args.crop_size,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    for epoch in range(curr_epoch, args.num_epochs):
        if local_rank == 0:
            print(f"EPOCH: {epoch + 1}/{args.num_epochs}")
            print("TRAINING")

        train_sampler.set_epoch(epoch)
        model_ddp.train()
        losses = []
        for img, _ in tqdm(train_loader, desc="Training"):
            img = img.to(device)
            with torch.enable_grad():
                recon, img = model_ddp(img)
                loss = loss_fn(recon, img)
                loss = reduce_mean(loss, nprocs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            if local_rank == 0:
                losses.append(loss.item())
        scheduler.step()

        if local_rank == 0:
            loss = np.mean(losses)
            hist_costs.append(loss)
            print(f"Loss: {loss:.4f}")
            print("TESTING")
            model.load_state_dict(model_ddp.module.state_dict())
            model.eval()
            preds, trues = [], []
            for img, label in tqdm(test_loader, desc="Testing"):
                img = img.to(device)
                with torch.no_grad():
                    recon, img = model(img)
                    # pred = 1. - score_fn(recon, img)
                    pred = score_fn(recon, img)
                preds.append(pred.cpu().numpy())
                trues.append(label.numpy())

            preds = np.concatenate(preds, axis=0)
            preds = (preds - preds.min()) / (preds.max() - preds.min()) if preds.min() < 0. or preds.max() > 1. else preds
            trues = np.concatenate(trues, axis=0)
            idx = rng.choice(len(preds), 5, replace=False)
            print(f"True: {trues[idx]}")
            print(f"Pred: {preds[idx]}")
            scores = {
                "auc_roc": roc_auc_score(trues, preds),
                "auc_pr": average_precision_score(trues, preds)
            }
            for metric, score in scores.items():
                hist_scores[metric].append(score)
                print(f"{metric}: Curr: {score:.4f}; Prev Best: {best_scores[metric]:.4f}")
                if score > best_scores[metric]:
                    best_scores[metric] = score
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{metric}_{score:.4f}.pth"))
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_{metric}.pth"))

            print("Saving checkpoint...")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "curr_epoch": epoch + 1,
                "hist_costs": hist_costs,
                "hist_scores": hist_scores,
                "best_scores": best_scores,
            }
            torch.save(checkpoint, checkpoint_path)
        torch.distributed.barrier()

        
if __name__ == "__main__":
    main()
