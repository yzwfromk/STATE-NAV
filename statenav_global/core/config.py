"""Single resolver for planning_config.yaml, shared by every statenav_global module and node.

Resolution order (same environment variable statenav_fsm.NavigationFSM honours):
  1. $STATENAV_PLANNING_CONFIG, when set (lets library users supply their own config without
     editing the installed package);
  2. the packaged default, statenav_global/configs/planning_config.yaml.
Dependencies: omegaconf only, so the core contract layer can use it.
"""
import os
from pathlib import Path

from omegaconf import OmegaConf

CONFIG_ENV_VAR = "STATENAV_PLANNING_CONFIG"
# Packaged default config (statenav_global/configs/planning_config.yaml).
_PACKAGED_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "planning_config.yaml"


def planning_config_path() -> Path:
    """Return the planning config path to load (not checked for existence; see load_planning_config)."""
    env_path = os.environ.get(CONFIG_ENV_VAR)
    return Path(env_path).expanduser().resolve() if env_path else _PACKAGED_CONFIG


def load_planning_config():
    """Load planning_config.yaml as a fresh OmegaConf object (one independent object per call)."""
    path = planning_config_path()
    if not path.is_file():
        source = f"${CONFIG_ENV_VAR}" if os.environ.get(CONFIG_ENV_VAR) else "packaged default"
        raise FileNotFoundError(f"planning config not found: {path} (from {source})")
    return OmegaConf.load(path)
