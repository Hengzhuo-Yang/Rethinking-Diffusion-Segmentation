import argparse
import csv
import json
from pathlib import Path

import torch

import logger
from metrics import sampling_major_vote_func
from preprocess_dataset.dataset import create_dataset
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from train_cDal_monu_and_lung import Posterior_Coefficients, sample_from_model
from utils import dev, set_random_seed_for_iterations
from release_validation import sha256_file, validate_best_checkpoint


def load_parameters(path):
    candidate = Path(path)
    if not candidate.is_file() and not candidate.is_absolute():
        candidate = Path(__file__).resolve().parent / candidate
    with candidate.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def namespace_from_dict(values):
    return argparse.Namespace(**values)


def normalize_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "netG_dict" in checkpoint:
        checkpoint = checkpoint["netG_dict"]
    if isinstance(checkpoint, dict) and all(isinstance(key, str) for key in checkpoint.keys()):
        if any(key.startswith("module.") for key in checkpoint.keys()):
            checkpoint = {key.replace("module.", "", 1): value for key, value in checkpoint.items()}
    return checkpoint


def write_summary_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate cDAL on multi-class 2D ACDC")
    parser.add_argument("--parameters", default="parameters_acdc.json")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_metadata", required=True)
    parser.add_argument("--data_dir", default="data_preprocessed/acdc_mt_unet_cascade_cdal_png")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["test", "testing"], default="testing")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--per_slice_csv", required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--major_vote_number", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_items", type=int, default=0)
    parser.add_argument("--metric_skip_empty_gt", type=lambda v: str(v).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--audit_mode", default=None)
    parser.add_argument("--save_visuals", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    cfg = load_parameters(cli.parameters)
    cfg.update(
        {
            "dataset": "acdc",
            "data_dir": cli.data_dir,
            "local_rank": cli.local_rank,
            "major_vote_number": cli.major_vote_number,
            "eval_batch_size": cli.eval_batch_size,
            "eval_max_items": cli.eval_max_items,
            "metric_skip_empty_gt": cli.metric_skip_empty_gt,
            "num_channels": 3,
            "num_channels_disc": 6,
            "save_visuals": cli.save_visuals,
        }
    )
    if cli.audit_mode is not None:
        cfg["audit_mode"] = cli.audit_mode
    args = namespace_from_dict(cfg)
    if int(args.num_channels) != 3 or int(args.num_channels_disc) != 6:
        raise ValueError("ACDC cDAL evaluation requires num_channels=3 and num_channels_disc=6")
    checkpoint_metadata = validate_best_checkpoint(
        cli.checkpoint_metadata,
        cli.model_path,
        "acdc",
        args.audit_mode,
    )

    output_folder = Path(cli.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if cli.overwrite:
        for target in [Path(cli.summary_csv), Path(cli.summary_json), Path(cli.per_slice_csv)]:
            if target.exists():
                target.unlink()

    logger.configure(dir=str(output_folder))
    device = dev(cli.local_rank)
    set_random_seed_for_iterations(int(getattr(args, "seed", 47)))

    dataset = create_dataset(
        data_dir=cli.data_dir,
        mode=cli.split,
        image_size=int(args.image_size),
        dataset_name="acdc",
        fold=int(getattr(args, "fold", 0)),
        manifest_path=cli.manifest,
    )

    netG = NCSNpp(args).to(device)
    checkpoint = torch.load(cli.model_path, map_location=device)
    netG.load_state_dict(normalize_state_dict(checkpoint), strict=True)
    netG.eval()
    pos_coeff = Posterior_Coefficients(args, device)

    miou, dice = sampling_major_vote_func(
        pos_coeff,
        sample_from_model,
        netG,
        output_folder,
        dataset,
        logger,
        None,
        args,
        device,
        metrics_csv=cli.per_slice_csv,
        summary_json=cli.summary_json,
    )
    summary = json.loads(Path(cli.summary_json).read_text(encoding="utf-8"))
    summary.update(
        {
            "model_path": str(cli.model_path),
            "checkpoint_metadata": str(cli.checkpoint_metadata),
            "checkpoint_selected_from": checkpoint_metadata["validation_split"],
            "checkpoint_loaded": True,
            "parameters": str(cli.parameters),
            "data_dir": str(cli.data_dir),
            "test_manifest": str(cli.manifest),
            "test_manifest_sha256": sha256_file(cli.manifest),
            "summary_csv": str(cli.summary_csv),
            "per_slice_csv": str(cli.per_slice_csv),
            "dice": dice,
            "miou": miou,
        }
    )
    Path(cli.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(cli.summary_csv, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
