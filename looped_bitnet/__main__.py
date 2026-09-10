"""Run with python -m looped_bitnet; long experiments are always explicit."""
import argparse
from dataclasses import replace
import json

from .config import Config
from .engine import evaluate_checkpoint, load_checkpoint, train
from .runtime import device_for, environment


def main():
    parser = argparse.ArgumentParser(description="Cyclic recurrent pointer chasing with ternary QAT")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("env", help="inspect Python/PyTorch/accelerators")
    training = commands.add_parser("train", help="fresh online training; --updates is the total target")
    training.add_argument("--config", help="JSON configuration; defaults to configs/base.json for new runs")
    training.add_argument("--out", required=True)
    training.add_argument("--resume", help="restore model, optimizer, data cursor and RNG")
    training.add_argument("--steps", type=int)
    training.add_argument("--updates", type=int)
    training.add_argument("--seed", type=int)
    training.add_argument("--float", action="store_true", help="use plain FP32 linear layers")
    evaluation = commands.add_parser("evaluate", help="evaluate a checkpoint with one or more step budgets")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--steps", nargs="+", type=int, default=[1, 4, 8, 16])
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--batch-size", type=int)
    evaluation.add_argument("--splits", nargs="+", choices=["validation", "in_distribution", "longer"])
    for command in (info, training, evaluation):
        command.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    device = device_for(args.device)
    if args.command == "env":
        print(json.dumps(environment(device), indent=2))
    elif args.command == "train":
        if args.resume:
            config = Config.from_dict(load_checkpoint(args.resume)["config"]) if not args.config else Config.load(args.config)
        else:
            config = Config.load(args.config or "configs/base.json")
        changes = {key: value for key, value in (("updates", args.updates), ("seed", args.seed)) if value is not None}
        config = replace(config, train=replace(config.train, **changes),
                         model=replace(config.model, **({"steps": args.steps} if args.steps is not None else {}),
                                       **({"quantized": False} if args.float else {})))
        train(config, args.out, device, args.resume)
    else:
        if any(step <= 0 for step in args.steps):
            parser.error("--steps must be positive")
        if args.batch_size is not None and args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        evaluate_checkpoint(args.checkpoint, args.steps, device, args.out, args.batch_size, args.splits)


if __name__ == "__main__":
    main()
