from .preprocess_bcm_data import preprocess_bcm_data
from .preprocess_dummy_data import preprocess_dummy_data

preprocess_data_map = {
    "preprocess_bcm_data": preprocess_bcm_data,
    "preprocess_dummy_data": preprocess_dummy_data
}
