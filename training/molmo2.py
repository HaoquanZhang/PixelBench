"""Instruction-tune Molmo2 with its official FSDP2 trainer and dynamic packer."""

import argparse
from pathlib import Path

from pixelbench.io import read_json, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="native Molmo2-4B-Pretrain step32000 directory")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data", required=True)
    parser.add_argument("--pixel-count", type=int, choices=[0, 1000, 2000, 4000, 8000], default=2000)
    parser.add_argument("--seed", choices=["s0", "s1", "s2", "s3"], default="s0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from launch_scripts.sft import get_model
    from olmo.data import data_loader
    from olmo.data.dataset import Dataset
    from olmo.data.dynamic_packer import PackingConfig
    from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
    from olmo.train.run_trainer import run_trainer
    from olmo.train.trainer_config import TrainConfig, FSDPConfig, BatchDivisor
    from olmo.util import prepare_torchrun_environment

    prepare_torchrun_environment()
    cfg = read_json("configs/molmo2.json")
    params, seeds = cfg["training"], cfg["seeds"][args.seed]
    rows = read_jsonl(args.data)
    if len(rows) != 500000 + args.pixel_count:
        raise ValueError("Molmo2 requires OV 500k plus the specified Pixel count")

    class PixelDataset(Dataset):
        def __len__(self):
            return len(rows)

        def get(self, item, rng):
            row = rows[item]
            return {"image": row["image"], "message_list": [
                {"question": row["prompt"], "answer": row["target"], "style": "user_qa"}]}

    # A single local JSONL dataset; the official loader handles shuffle and packing.
    dataset = PixelDataset()
    data_loader.get_dataset_by_name = lambda name, split: dataset
    model = get_model(args.checkpoint, "video")
    model.llm.tokenizer.identifier = args.tokenizer
    model.llm.compile = None
    model.mm_preprocessor.max_answer_len = params["max_answer_length"]
    model.mm_preprocessor.image.max_images = 1
    model.mm_preprocessor.image.high_res_max_crops = params["high_res_max_crops"]
    model.mm_preprocessor.video.max_frames = 16
    steps = cfg["steps"][str(args.pixel_count)]
    config = TrainConfig(
        run_name=Path(args.output).name, save_folder=args.output, seed=seeds["train"],
        model=model, initial_model_checkpoint=args.checkpoint,
        ft_llm=True, ft_vit=True, ft_connector=True,
        data=data_loader.DataLoaderConfig(
            dataset="pixelbench", split="train", shuffle=True, drop_last=True,
            sequence_length=params["sequence_length"], max_text_seq_len=None,
            num_workers=2, prefetch_factor=2, pin_memory=True, pad="to_max", seed=seeds["data"],
            packing=PackingConfig(buffer_size=48, image_weight=30, shortcut_max_len_images=False)),
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw, llm_learning_rate=params["llm_learning_rate"],
            vit_learning_rate=params["vision_learning_rate"], connector_learning_rate=params["vision_learning_rate"]),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal, connector_t_warmup=params["warmup_steps"],
            vit_t_warmup=params["warmup_steps"], llm_t_warmup=params["warmup_steps"],
            alpha_f=params["final_lr_multiplier"], warmup_min_lr=0.0),
        fsdp=FSDPConfig(fsdp2=True), global_train_batch_size=params["global_train_batch_size"],
        device_train_microbatch_size=1, max_duration=steps, stop_at=steps,
        max_grad_norm=params["gradient_clip"], precision=params["precision"],
        batch_divisor=BatchDivisor.global_batch, save_interval=steps,
        save_num_checkpoints_to_keep=1, save_final_optim=False,
        save_final_unsharded_checkpoint=False, wandb=None, compile=None, compile_loss=True,
        softmax_auxiliary_loss=True, softmax_auxiliary_loss_scale=1e-4,
        response_logits_only=True, evaluators=[], inf_evaluators=[], eval_interval=-1, inf_eval_interval=-1)
    run_trainer(config)


if __name__ == "__main__":
    main()
