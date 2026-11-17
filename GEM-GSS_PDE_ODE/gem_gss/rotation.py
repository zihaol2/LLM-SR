"""Frozen initial partition; a second sweep forces previously untested roots."""

import math
from .config import SETTINGS


def partition(ids, rng):
    ids = sorted(ids)
    rng.shuffle(ids)
    count = 10
    return [ids[i * len(ids) // count : (i + 1) * len(ids) // count] for i in range(count)]


def group_for(groups, number):
    return list(groups[(number - 1) % 10])


def quota(sampled):
    if SETTINGS.get('coverage_schedule')=='global_1pct':return math.ceil(len(sampled)/10)
    return math.ceil(len(sampled)*SETTINGS['minimum_test_fraction'])


def eligible(archive,sampled):
    if SETTINGS.get('coverage_schedule')!='global_1pct':return set(sampled)
    untested=set(sampled)-archive.tested_ids()
    # Keep the original preference for unexplored mechanisms, but do not
    # prescribe which untested IDs the LLM must choose or fill the whole group.
    return untested if len(untested)>=quota(sampled) else set(sampled)


def forced_for(archive, sampled, number):
    if SETTINGS.get('coverage_schedule')=='global_1pct':return []
    if number <= 10:
        return []
    tested = {mid for support in archive.attempted for mid in support}
    missing = sorted(set(sampled) - tested)
    if len(missing) > math.ceil(len(sampled) / 2):
        raise AssertionError("First-sweep coverage invariant was broken")
    return missing
