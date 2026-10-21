import dataclasses


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    functions_per_prompt: int = 3
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000
