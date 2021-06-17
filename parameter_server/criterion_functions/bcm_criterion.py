def bcm_criterion(rank, model):
    r"""
    A method that obtains a FairseqCriterion from the
    model attribute for training. The FairseqCriterion
    is obtained from the model attribute because it is
    built after building the model from a pickled FairseqConfig.
    Args:
        rank (int): worker rank
        model (FairseqEncoderModel): neural network model

    TODO: check to see if this can be moved to GPU?
    """
    return model.model_criterion
