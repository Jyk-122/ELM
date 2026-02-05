import argparse
import copy
import datetime
import json
import os
import sys
import time
import math
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import deepspeed

from models import ELMs
from util.datasets import SupervisedDataset
import util.lr_sched as lr_sched
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("ELM finetuing", add_help=False)
    # DeepSpeed parameters
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed ZeRO-2")
    parser.add_argument("--deepspeed_config", type=str, default="", help="Path to DeepSpeed config JSON")
    
    # Training parameters
    parser.add_argument("--batch_size", default=1, type=int, help="batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=10, type=int, help="epochs for training.")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--accum_iter", default=1, type=int, help="accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument("--print_freq", default=10, type=int, help="frequency of iterations to print logging infos")
    parser.add_argument("--save_freq", default=1, type=int, help="frequency of epochs to save checkpoints")
    parser.add_argument("--save_full_checkpoint", action="store_true", help="Enable DeepSpeed checkpoint saving")

    # Model parameters
    parser.add_argument("--hf_model_path", default="./llama", type=str, help="path to llama model checkpoints")
    parser.add_argument("--lora_model_path", default=None, type=str, help="path to lora model checkpoints")
    parser.add_argument("--model_save_name", type=str, default="Llama_dynamic", help="folder name to save your models")
    parser.add_argument("--max_batch_size", type=int, default=32, help="the maximum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=512, help="the maximum sequence length")
    parser.add_argument("--prune_interval_start_layer", type=int, default=1, help="")
    parser.add_argument("--prune_interval_layer_stats", type=str, default=None, help="")
    parser.add_argument("--prune_interval_len", type=int, default=8, help="")
    parser.add_argument("--lambda_lmloss", type=float, default=1, help="")
    parser.add_argument("--lambda_distill", type=float, default=1, help="")
    parser.add_argument("--fixed_point_wo_grad_min_iters", type=int, default=0, help="")
    parser.add_argument("--fixed_point_wo_grad_max_iters", type=int, default=8, help="")
    parser.add_argument("--fixed_point_with_grad_min_iters", type=int, default=1, help="")
    parser.add_argument("--fixed_point_with_grad_max_iters", type=int, default=8, help="")
    
    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument("--clip_grad", type=float, default=0.1, help="gradient norm clip (default: 0.1)")
    parser.add_argument("--lr", type=float, default=None, help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=1e-3, help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--min_lr", type=float, default=0.0, help="lower lr bound for cyclic schedulers that hit 0")
    parser.add_argument("--warmup_epochs", type=float, default=40, help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--dataset_path", default="./dataset", type=str, help="dataset path")
    parser.add_argument("--dataset_name", default="alpaca", type=str, help="dataset name")
    parser.add_argument("--unittest", default=0, type=int)
    parser.add_argument("--data_type", default="instruction", type=str, help="dataset type")
    parser.add_argument("--prompt_type", default="llama", type=str, help="prompt type")
    parser.add_argument("--output_dir", default="./output", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def _get_deepspeed_config(args):
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    world_size = misc.get_world_size()
    # Replace "auto" with actual values
    if ds_config.get("train_batch_size") == "auto":
        ds_config["train_batch_size"] = args.batch_size * args.accum_iter * world_size
    if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    if ds_config.get("gradient_accumulation_steps") == "auto":
        ds_config["gradient_accumulation_steps"] = args.accum_iter
    return ds_config


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (example, label, example_mask, label_mask) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            lambda_lmloss, lambda_distill = lr_sched.adjust_lambda(data_iter_step / len(data_loader) + epoch, args)
            iter_info = lr_sched.adjust_iter_info(data_iter_step / len(data_loader) + epoch, args)

        example      = example.to(device, non_blocking=True)
        label        = label.to(device, non_blocking=True)
        example_mask = example_mask.to(device, non_blocking=True)
        label_mask   = label_mask.to(device, non_blocking=True)
        
        loss_dict = model(example, label, example_mask, label_mask, iter_info)
        c_loss = loss_dict['c_loss']
        d_loss = loss_dict['d_loss']
        loss = lambda_lmloss * c_loss + lambda_distill * d_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        d_loss_value = d_loss.item()

        if not math.isfinite(loss_value):
            with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write("Loss is {} when training at example {}-{}.\n".format(loss_value, example, label))
            sys.exit(1)

        model.backward(loss)
        if (data_iter_step + 1) % accum_iter == 0:
            model.step()
        lr = model.get_lr()[0]

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(dloss=d_loss_value)
        metric_logger.update(lr=lr)

        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        d_loss_value_reduce = misc.all_reduce_mean(d_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            step = int(len(data_loader) * epoch + data_iter_step) // accum_iter
            log_writer.add_scalar("Train/c_loss", c_loss_value_reduce, step)
            log_writer.add_scalar("Train/d_loss", d_loss_value_reduce, step)
            log_writer.add_scalar("Train/lambda_lmloss", lambda_lmloss, step)
            log_writer.add_scalar("Train/lambda_distill", lambda_distill, step)
            log_writer.add_scalar("Train/lr", lr, step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = SupervisedDataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        partition="train",
        data_type=args.data_type,
        prompt_type=args.prompt_type,
        tokenizer_path=args.hf_model_path,
        max_len=args.max_seq_len,
        unittest=args.unittest,
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    save_folder_name = f"{args.dataset_name}/{args.model_save_name}_{time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))}"

    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, save_folder_name)
        args.log_dir = os.path.join(args.output_dir, "log")
        args.ckpt_dir = os.path.join(args.output_dir, "ckpt")

        if global_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.log_dir, exist_ok=True)
            os.makedirs(args.ckpt_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
            with open(os.path.join(args.output_dir, "params.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
        else:
            log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = ELMs(args)

    if global_rank == 0:
        if args.output_dir:
            with open(os.path.join(args.output_dir, "model_args.json"), "w") as f:
                json.dump(vars(model.params), f, indent=4)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    ds_config = _get_deepspeed_config(args)
    # weight decay: 0 for norm/bias (same as current logic)
    param_groups_with_decay = []
    param_groups_wo_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            param_groups_wo_decay.append(p)
        else:
            param_groups_with_decay.append(p)
    param_groups = [
        {"params": param_groups_with_decay, "weight_decay": args.weight_decay},
        {"params": param_groups_wo_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        model_parameters=model.parameters(),
    )
    model = model_engine
    loss_scaler = None
    
    misc.load_model_from_ds(args, model)

    print(f"Training on {args.dataset_name} for {args.epochs} epochs, starts at epoch = {args.start_epoch}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        if args.ckpt_dir and ((epoch + 1) % (args.save_freq) == 0 or epoch + 1 == args.epochs):
            tag = f"epoch_{epoch}"
            os.makedirs(os.path.join(args.ckpt_dir, tag), exist_ok=True)
            
            if args.save_full_checkpoint:
                model.save_checkpoint(args.ckpt_dir, tag, client_state={"epoch": epoch})
            
            misc.save_model(
                args=args,
                tag=tag,
                epoch=epoch,
                model=model,
                model_without_ddp=model_without_ddp,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.ckpt_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    
    main(args)
