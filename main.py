# main.py
from data.registry   import DATASET_REGISTRY, LOADER_REGISTRY
from trainer.registry import TRAINER_REGISTRY
from models.registry  import MODEL_REGISTRY
from omegaconf import OmegaConf

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
    # base_config = OmegaConf.load("config/CISFNO.yaml")
    
    # new_join = {
        
    # }
    # config = OmegaConf.merge(base_config, OmegaConf.create(new_join))
    exp_ = "exp13"
    train_list = ["config/CISFNO1.yaml","config/CISFNO1.yaml","config/CISFNO3.yaml", "config/CISFNO1_1.yaml", "config/CISFNO2_1.yaml", "config/CISFNO3_1.yaml"]
    train_list = ["config/CISFNO.yaml"]
    train_list = ["config/CIPFNO.yaml"]
    train_list = ["checkpoints/exp11/20260408_124446/config.yaml"]
    new_Remarks = f"{exp_}: 并联结构，loss=pixel_l2(skip, gt) + pixel_l2(up + skip, gt)" 
    new_save_path = f"checkpoints/{exp_}"
    model_type = "CIPFNO"
    # lr_min = 1e-5
    
    epochs = 400
    for path in train_list:
        config = OmegaConf.load(path)
        # config.trainer.type = "CIPtrainer"
        # config["备注"] = new_Remarks
        # config.loader.val_batch_size = 1
        # config.model.type = model_type
        config.trainer.device = "cuda:0"
        # config.trainer.save_path = new_save_path
        # config.trainer.loss.high_freq_weight = 1.0
        # config.trainer.loss.low_freq_weight = 0.0
        # config.trainer.loss.low_freq_ratio = 0.4
        # config.trainer.scheduler.lr_min = lr_min
        # config.trainer.epochs = epochs
        main(config)