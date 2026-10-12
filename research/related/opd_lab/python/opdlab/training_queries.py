"""Frozen, task-balanced queries shared by controlled on-policy training arms."""
from collections import defaultdict
import json
import random

from .data import records_digest
from .io import stable_seed


GROUP_SCHEDULE = "balanced_groups_shuffled_each_pass_v1"
STRATUM_SCHEDULE = "balanced_family_template_shuffled_each_pass_v2"
PAIRED_TIME_SCHEDULE = "balanced_family_template_paired_time_v3"


def selection_stratum(record):
    """Same outcome-independent key as semantic batch selection."""
    family = sorted(c['id'] for c in record['constraints'] if c['kind'] != 'ocr')
    return json.dumps([family, record['group']], separators=(',', ':'))


def make_training_manifest(config, records, world_size=1, *, schedule=GROUP_SCHEDULE):
    if schedule not in (GROUP_SCHEDULE, STRATUM_SCHEDULE, PAIRED_TIME_SCHEDULE):
        raise ValueError('Unknown training query schedule')
    stratified = schedule in (STRATUM_SCHEDULE, PAIRED_TIME_SCHEDULE)
    if stratified:
        if world_size != 1 or config.accumulation != config.queries_per_rank:
            raise ValueError('Family/template schedule requires one process and one complete accumulation batch')
        if not records or len({r['id'] for r in records}) != len(records):
            raise ValueError('Require nonempty unique training records')
    groups = defaultdict(list)
    for record in records:
        group = selection_stratum(record) if stratified else record['group']
        groups[group].append(record["id"])
    batch = world_size * config.queries_per_rank
    if (len({len(ids) for ids in groups.values()}) != 1 or batch % len(groups)
            or len(records) % batch):
        raise ValueError("Balanced manifest requires equal groups and a global batch dividing the dataset")
    if stratified and batch // len(groups) < 2:
        raise ValueError('Each selection stratum needs at least two candidates per accumulation batch')
    if schedule == PAIRED_TIME_SCHEDULE and batch // len(groups) != 2:
        raise ValueError('Paired-time schedule requires exactly two queries per family/template per batch')
    header = {"format": 1, "schedule": schedule,
              "records_digest": records_digest(records), "seed": config.seed,
              "world_size": world_size, "queries_per_rank": config.queries_per_rank,
              "outer_epochs": config.outer_epochs, "steps": config.steps,
              "timestep_fraction": config.timestep_fraction}
    queries, cycle = [], 0
    total = batch * config.outer_epochs
    while len(queries) < total:
        shuffled = {}
        for group, ids in sorted(groups.items()):
            shuffled[group] = sorted(ids)
            random.Random(stable_seed(config.seed, "train_order", cycle, group)).shuffle(shuffled[group])
        order = sorted(groups)
        random.Random(stable_seed(config.seed, "group_order", cycle)).shuffle(order)
        for offset in range(len(next(iter(groups.values())))):
            for group in order:
                ordinal = len(queries)
                if ordinal == total:
                    break
                epoch = ordinal // batch + 1
                index = stable_seed(config.seed, "query_time", ordinal) % int(config.steps * config.timestep_fraction)
                if schedule == PAIRED_TIME_SCHEDULE:
                    # Share only time, never noise or prompt. Each full block of
                    # training-time-count epochs visits every index per stratum.
                    times = list(range(int(config.steps * config.timestep_fraction)))
                    block, position = divmod(epoch - 1, len(times))
                    random.Random(stable_seed(config.seed, "paired_query_time", group, block)).shuffle(times)
                    index = times[position]
                queries.append({"global_query_id": ordinal, "outer_epoch": epoch,
                                "id": shuffled[group][offset],
                                "seed": stable_seed(config.seed, "train_query", ordinal),
                                "query_index": index})
            if len(queries) == total:
                break
        cycle += 1
    return {**header, "queries": queries}


def validate_training_manifest(manifest, config, records, world_size):
    if manifest != make_training_manifest(config, records, world_size, schedule=manifest.get('schedule')):
        raise ValueError("Training query manifest does not match its data, budget, or deterministic schedule")
