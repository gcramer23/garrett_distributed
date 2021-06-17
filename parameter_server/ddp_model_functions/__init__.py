from .basic_ddp_model import basic_ddp_model
from .bcm_ddp_model import bcm_ddp_model

create_ddp_model_map = {
    "basic_ddp_model": basic_ddp_model,
    "bcm_ddp_model": bcm_ddp_model
}
