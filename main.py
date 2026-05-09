#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
import importlib
import numpy as np
from PIL import Image
import torch
from diffbir.utils.common import instantiate_from_config


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _get_section(cfg, *keys):
    current = cfg
    for key in keys:
        if current is None:
            return None
        if key not in current:
            return None
        current = current[key]
    return current


def _latest_checkpoint(ckpt_dir: str):
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    candidates = sorted(Path(ckpt_dir).glob('*.pt'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return str(candidates[-1])


def _resolve_checkpoint(test_cfg, train_cfg):
    for key in ['ckpt', 'checkpoint', 'ckpt_path']:
        value = test_cfg.get(key)
        if value:
            return value

    train_resume = train_cfg.get('train', {}).get('resume') if isinstance(train_cfg.get('train', {}), dict) else None
    if train_resume:
        return train_resume

    exp_dir = train_cfg.get('train', {}).get('exp_dir') if isinstance(train_cfg.get('train', {}), dict) else None
    if exp_dir:
        latest = _latest_checkpoint(os.path.join(exp_dir, 'checkpoints'))
        if latest:
            return latest
    return None


def call_train_stage(stage: int, config_path: str):
    if stage == 1:
        mod = importlib.import_module('train_stage1')
    elif stage == 2:
        mod = importlib.import_module('train_stage2')
    else:
        raise ValueError('stage must be 1 or 2')
    args = Namespace(config=config_path)
    mod.main(args)


def _save_tensor_image(tensor: torch.Tensor, save_path: str):
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    array = (array * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(save_path)


def _run_stage1_test(cfg, train_cfg_path: str):
    test_cfg = _get_section(cfg, 'experiment', 'test') or _get_section(cfg, 'test')
    train_cfg = OmegaConf.load(train_cfg_path)
    input_dir = test_cfg.get('input_dir')
    output_dir = test_cfg.get('output_dir')
    gt_dir = test_cfg.get('gt_dir')
    dataset_path = test_cfg.get('dataset_path')
    save_dir = test_cfg.get('save_dir')
    if not input_dir or not output_dir:
        raise ValueError('test config must provide input_dir and output_dir')

    ckpt_path = _resolve_checkpoint(test_cfg, train_cfg)
    if not ckpt_path:
        raise ValueError('cannot find a checkpoint; set test.ckpt or train.resume, or save checkpoints under exp_dir/checkpoints')

    model_cfg = train_cfg.model.swinir
    model = instantiate_from_config(model_cfg)
    from diffbir.utils.common import safe_torch_load
    state_dict_loaded = safe_torch_load(ckpt_path, map_location='cpu')
    state_dict = state_dict_loaded['state_dict'] if isinstance(state_dict_loaded, dict) and 'state_dict' in state_dict_loaded else state_dict_loaded
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    device = test_cfg.get('device', 'cuda')
    precision = test_cfg.get('precision', 'fp16')
    model.eval().to(device)

    os.makedirs(output_dir, exist_ok=True)
    auto_cast_type = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }[precision]

    img_files = sorted(
        [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    )
    for file_name in img_files:
        file_path = os.path.join(input_dir, file_name)
        lq = Image.open(file_path).convert('L')
        lq = np.array(lq, dtype=np.float32) / 255.0
        lq = np.repeat(lq[..., None], 3, axis=2)
        lq_tensor = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type=device if device != 'mps' else 'cpu', dtype=auto_cast_type, enabled=(device != 'cpu')):
            pred = model(lq_tensor)
        _save_tensor_image(pred[0], os.path.join(output_dir, file_name))

    if gt_dir and dataset_path and save_dir:
        blind_eval = importlib.import_module('diffbir.eval.blind_eval')
        eval_cfg = test_cfg.get('blind_eval', {})
        test_mask_csv = eval_cfg.get('test_mask_csv') if isinstance(eval_cfg, dict) else None
        blind_eval.test_quantitative_result(gt_dir, output_dir, dataset_path, save_dir, test_mask_csv)


def call_test(cfg):
    # allow stage-specific `test` sections; don't require a top-level `experiment.test` or `test`
    test_cfg = _get_section(cfg, 'experiment', 'test') or _get_section(cfg, 'test')

    experiment_cfg = _get_section(cfg, 'experiment') or cfg
    stage = int(experiment_cfg.get('stage', cfg.get('stage', 1)))
    stage_cfg = experiment_cfg.get(f'stage{stage}') if hasattr(experiment_cfg, 'get') else None
    if stage_cfg is not None:
        test_cfg = stage_cfg.get('test', test_cfg)
    train_cfg_path = experiment_cfg.get('train_config_path') or experiment_cfg.get('train_cfg_path')
    if stage_cfg is not None:
        train_cfg_path = stage_cfg.get('train_config_path', train_cfg_path)
    if not train_cfg_path and 'train' in cfg:
        train_cfg_path = cfg.train.get('config_path')
    if not train_cfg_path:
        train_cfg_path = 'configs/train/blind_train_stage1.yaml'

    if stage == 1:
        _run_stage1_test(cfg, train_cfg_path)
        return

    input_dir = test_cfg.get('input_dir')
    output_dir = test_cfg.get('output_dir')
    if not input_dir or not output_dir:
        raise ValueError('test config must provide input_dir and output_dir')

    inf_mod = importlib.import_module('inference')

    argv = [sys.argv[0], '--input', input_dir, '--output', output_dir]
    for key in ['device', 'n_samples', 'version', 'ckpt', 'train_cfg', 'task', 'sampler', 'steps', 'start_point_type', 'cleaner_tiled', 'vae_encoder_tiled', 'vae_decoder_tiled', 'cldm_tiled', 'captioner', 'cfg_scale', 'strength', 'batch_size', 'precision', 'llava_bit']:
        value = test_cfg.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f'--{key}')
        else:
            argv += [f'--{key}', str(value)]

    old_argv = sys.argv
    try:
        sys.argv = argv
        inf_mod.main()
    finally:
        sys.argv = old_argv

    eval_cfg = test_cfg.get('blind_eval') or _get_section(cfg, 'experiment', 'blind_eval')
    if eval_cfg is not None:
        gt_dir = eval_cfg.get('gt_dir')
        dataset_path = eval_cfg.get('dataset_path')
        save_dir = eval_cfg.get('save_dir')
        if gt_dir and dataset_path and save_dir:
            blind_eval = importlib.import_module('diffbir.eval.blind_eval')
            test_mask_csv = eval_cfg.get('test_mask_csv') if isinstance(eval_cfg, dict) else None
            blind_eval.test_quantitative_result(gt_dir, output_dir, dataset_path, save_dir, test_mask_csv)


def main():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--config_path', required=True, help='Path to experiment config (YAML)')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    experiment_cfg = _get_section(cfg, 'experiment') or cfg
    stage = experiment_cfg.get('stage', cfg.get('stage', 1))
    stage_cfg = experiment_cfg.get(f'stage{stage}') if hasattr(experiment_cfg, 'get') else None
    train_cfg_path = None
    if stage_cfg is not None:
        train_cfg_path = stage_cfg.get('train_config_path')
    if not train_cfg_path:
        train_cfg_path = experiment_cfg.get('train_config_path')
    if not train_cfg_path:
        train_cfg_path = experiment_cfg.get('train_cfg_path')
    if not train_cfg_path and 'train' in cfg:
        train_cfg_path = cfg.train.get('config_path')
    if not train_cfg_path:
        train_cfg_path = args.config_path

    if args.train:
        print(f"Starting training stage {stage} with config {train_cfg_path}")
        call_train_stage(int(stage), train_cfg_path)
    else:
        print(f"Starting testing using config {args.config_path}")
        call_test(cfg)


if __name__ == '__main__':
    main()
