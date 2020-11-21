# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs"""

# python scripts/aggregator.py -r result/CAModel-all-4e-nict-coref-ocz-noun

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from functools import reduce

import numpy as np
from ordered_set import OrderedSet
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import FileWriter
from torch.utils.tensorboard.summary import Summary
from torch.utils.tensorboard.writer import Event


def extract(paths: List[Path]) -> Dict[str, tuple]:
    event_accumulators: List[EventAccumulator] = [EventAccumulator(str(path)).Reload() for path in paths]

    all_scalar_tags: List[OrderedSet[str]] = [OrderedSet(accum.Tags()['scalars']) for accum in event_accumulators]
    tags: OrderedSet[str] = reduce(lambda x, y: x & y, all_scalar_tags)

    return {tag: _extract_tag(event_accumulators, tag) for tag in tags}


def _extract_tag(event_accumulators: List[EventAccumulator],
                 tag: str
                 ) -> Tuple[List[float], tuple, List[List[float]]]:
    all_scalar_events: List[List[Event]] = [accum.Scalars(tag) for accum in event_accumulators]

    wall_times: List[float] = list(np.mean([[event.wall_time for event in events] for events in all_scalar_events],
                                           axis=0))

    all_steps: List[tuple] = [tuple(event.step for event in events) for events in all_scalar_events]
    assert len(set(all_steps)) == 1, \
        'For scalar {} the step numbering or count doesn\'t match. Step count for all runs: {}'.format(
            tag, [len(steps) for steps in all_steps])
    steps: tuple = all_steps[0]

    all_values: List[List[float]] = [[event.value for event in events] for events in all_scalar_events]

    return wall_times, steps, all_values


def write_summary(base_dir: Path, aggregations_per_tag) -> None:
    writer = FileWriter(base_dir)

    for tag, (steps, wall_times, aggregations) in aggregations_per_tag.items():
        for wall_time, step, aggregation in zip(steps, wall_times, aggregations):
            summary = Summary(value=[Summary.Value(tag=tag, simple_value=aggregation)])
            scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)

        writer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', '-r', type=str, required=True, help='main path for tensorboard files')
    parser.add_argument('--aggr-name', type=str, default='aggregates',
                        help='name of directory where aggregated summaries are output')
    parser.add_argument('--operations', '--ops', choices=['mean', 'min', 'max', 'median', 'std', 'var'],
                        default=['mean', 'min', 'max'], nargs='*',
                        help='operations to aggregate summaries')
    args = parser.parse_args()

    base_dir = Path(args.result)
    if not base_dir.exists():
        raise argparse.ArgumentTypeError(f'Parameter {base_dir} is not a valid path')

    tfevents_paths = list(base_dir.glob('*/events.out.tfevents.*'))
    if not tfevents_paths:
        raise ValueError(f'No tfevents file found in {base_dir}/*/')

    print(f'Started aggregation {base_dir.name}')

    extracts = extract(tfevents_paths)
    for op_name in args.operations:
        op = getattr(np, op_name)
        summary_dir = base_dir / args.aggr_name / op_name
        aggregations_per_tag = {tag: (steps, wall_times, op(values, axis=0))
                                for tag, (steps, wall_times, values) in extracts.items()}
        write_summary(summary_dir, aggregations_per_tag)

    print(f'Ended aggregation {base_dir.name}')


if __name__ == '__main__':
    main()
