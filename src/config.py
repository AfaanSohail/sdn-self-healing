"""Fixed experimental specification; changes here require rerunning every method."""
from dataclasses import dataclass

FAULTS = ('outage', 'congestion', 'loss', 'delay')
DEVELOPMENT = ('normal', 'burst', *FAULTS, 'concurrent')
SCENARIOS = (*DEVELOPMENT, 'correlated', 'cascading', 'switch', 'infeasible')
LOADS = ('low', 'medium', 'high')
LOAD_BASE = {'low': 27.0, 'medium': 53.0, 'high': 72.0}
METHODS = ('B0_no_recovery', 'B1_rule', 'B2_greedy', 'Full_DQN', 'A1_no_diagnosis', 'A2_no_shield', 'B3_GA', 'B4_PSO', 'A4_predictive', 'A5_no_topology', 'A6_no_SHAP', 'Oracle')
TRAINING_SEEDS = (11, 22, 33, 44, 55)

@dataclass(frozen=True)
class Settings:
    episode_seconds: int = 180
    inject_at: int = 60
    clear_at: int = 120
    history: int = 10
    nominal_capacity: float = 100.0
    nominal_link_delay_ms: float = 2.0
    rtt_sla_ms: float = 20.0
    loss_sla: float = .01
    goodput_ratio_sla: float = .90
    cooldown: int = 5
    stable_period: int = 5
    capacity_factor: float = 1.2
    train_episodes_per_group: int = 60
    val_episodes_per_group: int = 20
    test_episodes_per_group: int = 20
    # Executed final evaluation target.
    blocks_per_stratum: int = 30
    # Executed cap; Phase 2 specified an upper bound of 500,000.
    training_steps_per_seed: int = 50000

SET = Settings()
