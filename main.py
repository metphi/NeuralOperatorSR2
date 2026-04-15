# main.py
from data.registry   import DATASET_REGISTRY, LOADER_REGISTRY
from trainer.registry import TRAINER_REGISTRY
from models.registry  import MODEL_REGISTRY
from omegaconf import OmegaConf
import argparse

# 自动import所有实现类，触发装饰器注册
import data.datasets 
import trainer.trainers 
import data.loaders
import models.models

def main(config):
    train_dataset = DATASET_REGISTRY.build(config.data.type, config, split='train')
    val_dataset   = DATASET_REGISTRY.build(config.data.type, config, split='val')

    train_loaders = LOADER_REGISTRY.build(
        config.loader.type,
        dataset=train_dataset,
        config=config
    )
    val_loader = LOADER_REGISTRY.build(
        config.loader.type,
        dataset=val_dataset,
        config=config
    )
    loaders = {"train": train_loaders, "val": val_loader}

    model   = MODEL_REGISTRY.build(config.model.type, config)
    trainer = TRAINER_REGISTRY.build(config.trainer.type, model, loaders, config)
    trainer.train()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:0" or "cpu").')
    parser.add_argument("--save_path", type=str, help="Path to save checkpoints and logs(overrides config if provided).")
    parser.add_argument("--data_path", type=str, help="Path to the dataset (overrides config if provided).")
    parser.add_argument('-m', "--remarks", type=str, default="", help="Additional remarks for the experiment.")
    parser.add_argument("--lr", type=float, help="Learning rate for training (overrides config if provided).")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config if provided).")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility (overrides config if provided).")
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.trainer.device = args.device
    config.remarks = args.remarks
    if args.lr is not None:
        config.trainer.lr = args.lr
    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.seed is not None:
        config.trainer.seed = args.seed
        config.data.seed = args.seed
    if args.save_path is not None:
        config.trainer.save_path = args.save_path
    if args.data_path is not None:
        config.data.data_path = args.data_path
        
        
    main(config)
    