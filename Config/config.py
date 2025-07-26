from dataclasses import dataclass

@dataclass
class TaskConfig:
    experiment_name: str
    max_interactions: int
    sample_size: int
    env_type: str
    run_with_experience: bool
    create_exp: bool
    best_at: int



