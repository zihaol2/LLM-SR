"""Frozen initial partition; a second sweep forces previously untested roots."""

import math


def partition(ids, rng):
    ids = sorted(ids)
    rng.shuffle(ids)
    return [ids[i * len(ids) // 10 : (i + 1) * len(ids) // 10] for i in range(10)]


def group_for(groups, number):
    return list(groups[(number - 1) % 10])


def forced_for(archive, sampled, number):
    if number <= 10:
        return []
    tested = {mid for support in archive.attempted for mid in support}
    missing = sorted(set(sampled) - tested)
    if len(missing) > math.ceil(len(sampled) / 2):
        raise AssertionError("First-sweep coverage invariant was broken")
    return missing
