# utils/criterion.py
import torch


class CriterionGroup:
    """
    统一管理多个 loss/metric criterion
    区分两种角色：
      - backward_loss: 参与反向传播（只能有一个标量）
      - monitor_losses: 只记录不反向（可以有多个）
    """
    def __init__(self):
        self._backward: dict  = {}   # 参与backward的loss，通常只有1个
        self._monitor:  dict  = {}   # 只用于log/监控的metric

    def add_backward(self, name: str, fn, weight: float = 1.0):
        self._backward[name] = {"fn": fn, "weight": weight}
        return self

    def add_monitor(self, name: str, fn):
        self._monitor[name] = {"fn": fn}
        return self

    def compute(self, preds, targets) -> dict:
        """
        返回:
          - 'loss':         最终用于backward的标量（加权求和）
          - 'loss_detail':  每个backward loss的分项值（用于log）
          - 'metrics':      所有monitor metric的值
        """
        results      = {}
        total_loss   = 0.0
        loss_detail  = {}

        # 计算 backward losses（加权求和）
        for name, item in self._backward.items():
            val = item["fn"](preds, targets)
            weighted = item["weight"] * val
            total_loss        += weighted
            loss_detail[name]  = val.item()

        # 计算 monitor metrics（不进计算图）
        with torch.no_grad():
            for name, item in self._monitor.items():
                results[name] = item["fn"](preds, targets).item()

        results["loss"]        = total_loss    # tensor，用于 .backward()
        results["loss_detail"] = loss_detail   # dict of floats，用于 log

        return results