import pickle

from examples.roberta.commonsense_qa.commonsense_qa_task import \
    CommonsenseQATask
from fairseq import tasks
import os


class BertCommonsenseModel:

    def __new__(cls, cfg_path="bcm_cfg.pkl"):
        os.chdir("/data/home/gcramer")
        with open(cfg_path, 'rb') as input:
            cfg = pickle.load(input)
        task = tasks.setup_task(cfg.task)
        model = task.build_model(cfg.model)
        criterion = task.build_criterion(cfg.criterion)
        setattr(model, "model_criterion", criterion)
        return model
