__version__ = "0.0.1"

# Import subpackages
# NOTE: the FSM moved out to the standalone `statenav_fsm` package so that repos without
# statenav_global (IsaacLab, elevation_mapping) can use it; import it directly:
#     from statenav_fsm.NavigationFSM import NavigationFSM, FSMState
from . import planners
from . import world_model
from . import utility
from . import inference

# Import all classes from subpackages (respects their __all__ definitions)
from .world_model import *
from .planners import *
from .utility import *
from .inference import *

# Automatically combine __all__ from subpackages
# This way, changes in subpackages automatically propagate here
__all__ = [
    # Subpackages
    "planners",
    "world_model",
    "utility",
    "inference",
] + world_model.__all__ + planners.__all__ + utility.__all__ + inference.__all__