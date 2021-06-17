from torch.nn.parallel import DistributedDataParallel as DDP

def bcm_ddp_model(self, rank, model, process_group, HookState, hook):
    r"""
    A method that creates a ddp_model and hook_state objects
    for the trainer. Bert training requires access to the
    classification_heads dictionary, and get_targets function.
    The DDP model does not provide access to those
    attributes. The method sets the required attributes.
    Args:
        rank (int): worker rank
        model (FairseqEncoderModel): neural network model
        process_group (ProcessGroup): distributed process group
        HookState (class): class that will be used to keep tracking of state
            during training.
        hook (function): ddp communication hook
    """
    ddp_model = DDP(
        model, device_ids=[rank], process_group=process_group, find_unused_parameters=True
    )
    setattr(ddp_model, "classification_heads", model.classification_heads)
    setattr(ddp_model, "get_targets", model.get_targets)
    hook_state = HookState(self, process_group)
    ddp_model.register_comm_hook(hook_state, hook)
    return ddp_model, hook_state
