"""Benchmark batched inference using an already trained RealPhe model.

Run from the repository root after the full phenotyping pipeline has produced
processed evaluation data and trained fold-specific models.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from realphe.pipe.transformer import transform
from realphe.task.phenotyping.baseline.load import pipeline_single_load
from realphe.task.phenotyping.baseline.setup import get_eval_set
from realphe.task.phenotyping.baseline.solution import get_dataset
from realphe.task.phenotyping.setup.dataset import packed_sample_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--warmup-batches', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def make_loader(dataset, batch_size: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=packed_sample_collate_fn,
    )


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError('--repeats must be at least 1.')
    if args.warmup_batches < 0:
        raise ValueError('--warmup-batches cannot be negative.')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    pin_memory = device.type == 'cuda'

    params = OmegaConf.load('params.yaml').phenotyping
    _, _, cohort_test, signals_test = get_eval_set(params, 'baseline')

    model, cohort_transformers, signals_transformers = pipeline_single_load(
        args.seed,
        args.fold,
    )

    # Initial loading and preprocessing are deliberately excluded from timing.
    cohort_test = transform(cohort_test, cohort_transformers)
    signals_test = transform(signals_test, signals_transformers)
    dataset = get_dataset(cohort_test, signals_test, shuffle=False)

    model = model.to(device)
    model.eval()

    if args.warmup_batches > 0:
        with torch.inference_mode():
            for batch_idx, (x_batch, _) in enumerate(make_loader(dataset, args.batch_size, pin_memory)):
                x_batch = x_batch.to(device, non_blocking=True)
                model(x_batch)
                if batch_idx + 1 >= args.warmup_batches:
                    break

        if device.type == 'cuda':
            torch.cuda.synchronize()

    n_stays = 0
    n_steps = 0
    with torch.inference_mode():
        for x_batch, y_batch in make_loader(dataset, args.batch_size, pin_memory):
            x_batch = x_batch.to(device, non_blocking=True)
            prediction = model(x_batch)
            n_stays += len(y_batch)
            n_steps += prediction.data.shape[0]

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # The timed region includes batch collation and host-to-device transfer.
    # Initial dataset loading and preprocessing above are excluded.
    times = []
    for _ in range(args.repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.inference_mode():
            for x_batch, _ in make_loader(dataset, args.batch_size, pin_memory):
                x_batch = x_batch.to(device, non_blocking=True)
                model(x_batch)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    median_time = statistics.median(times)

    print(f'Device:                  {device}')
    if device.type == 'cuda':
        print(f'GPU:                     {torch.cuda.get_device_name(device)}')
    print(f'ICU stays:               {n_stays:,}')
    print(f'Valid time steps:        {n_steps:,}')
    print(f'Batch size:              {args.batch_size}')
    print(f'Repeated runs:           {args.repeats}')
    print(f'Median prediction time:  {median_time:.4f} s')
    print(f'Range:                   {min(times):.4f}-{max(times):.4f} s')
    print(f'Median throughput:       {n_steps / median_time:,.1f} time steps/s')
    print(f'Amortized cost per stay: {1e3 * median_time / n_stays:.4f} ms')
    print(f'Amortized cost per step: {1e3 * median_time / n_steps:.6f} ms')


if __name__ == '__main__':
    main()
