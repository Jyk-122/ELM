import math
import sys
import numpy as np


def cos_anneal(e0, e1, t0, t1, e):
    """ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1]"""
    theta = np.clip((e - e0) / (e1 - e0), 0, 1) * math.pi / 2
    alpha = 1 - math.cos(theta)
    return (1 - alpha) * t0 + alpha * t1


def sin_anneal(e0, e1, t0, t1, e):
    """ramp from (e0, t0) -> (e1, t1) through a sine schedule based on e \in [e0, e1]"""
    theta = np.clip((e - e0) / (e1 - e0), 0, 1) * math.pi / 2
    alpha = math.sin(theta)
    return (1 - alpha) * t0 + alpha * t1



def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_lambda(epoch, args):
    lmloss_start_epoch = args.epochs / 2
    if epoch < lmloss_start_epoch:
        return 0, args.lambda_distill
    else:
        return sin_anneal(lmloss_start_epoch, args.epochs, 0, args.lambda_lmloss, epoch), args.lambda_distill

def adjust_iter_info(epoch, args):
    iter_info = {
        "fixed_point_wo_grad_min_iters": args.fixed_point_wo_grad_min_iters,
        "fixed_point_wo_grad_max_iters": args.fixed_point_wo_grad_max_iters,
        "fixed_point_with_grad_min_iters": args.fixed_point_with_grad_min_iters,
        "fixed_point_with_grad_max_iters": args.fixed_point_with_grad_max_iters,
    }
    
    return iter_info