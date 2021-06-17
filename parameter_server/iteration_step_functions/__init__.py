from .basic_iteration_step import basic_iteration_step
from .bcm_iteration_step import bcm_iteration_step

iteration_step_map = {
    "basic_iteration_step": basic_iteration_step,
    "bcm_iteration_step": bcm_iteration_step
}
