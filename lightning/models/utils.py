from inspect import getmro, signature
from typing import Optional, Type, Dict
from warnings import warn
from pytorch_metric_learning.losses import CrossBatchMemory, ContrastiveLoss
from pytorch_metric_learning.miners import BaseMiner
from torch_metric_learning.miners import create_pa_miner, create_dummy_miner, passthrough_miner, prism
from torch_metric_learning.noise_reducers.memory_bank import MemoryBank
from torch_metric_learning.noise_reducers.sample_rejection.rejection_base import RejectionStrategy
from torch_metric_learning.noise_reducers.sample_rejection.rejection_criteria import RejectionCriterion, CombinedCriteria
from torch_metric_learning.reducers import create_noisy_weighted_reducer
from utils import class_from_string


def construct_miner(miner_class, miner_kwargs, distance) -> BaseMiner:
    if miner_kwargs is None:
        miner_kwargs = {}

    # Override None with a passthrough miner (which does nothing), in order to avoid extra logic during training,
    # i.e., if miner == None, etc
    if miner_class is None:
        miner_class = passthrough_miner.PassThroughMiner

    return miner_class(distance=distance, **miner_kwargs)


def construct_rejection_criteria(strategy_kwargs):
    rejection_criteria = strategy_kwargs.get('rejection_criteria')

    if rejection_criteria is None:
        return

    if isinstance(rejection_criteria, RejectionCriterion):
        # This case is when the rejection_criteria already exists, e.g. saved in the model state and loaded afterwards
        # probably the strategy is also saved therefore may never reach here
        return rejection_criteria
    elif isinstance(rejection_criteria, str):
        rejection_criteria_class = class_from_string(rejection_criteria)
    elif isinstance(rejection_criteria, type(RejectionCriterion)) \
            or isinstance(rejection_criteria, type(CombinedCriteria)):
        rejection_criteria_class = rejection_criteria
    else:
        raise ValueError('Rejection criteria instance can be constructed either by string representation of the full '
                         'class path or by class type')

    try:
        rejection_criteria_kwargs = strategy_kwargs.pop('rejection_criteria_kwargs')
        if not isinstance(rejection_criteria_kwargs, dict):
            raise ValueError('Invalid strategy kwargs type. Should be dict.')
    except KeyError:
        rejection_criteria_kwargs = {}

    return rejection_criteria_class(**rejection_criteria_kwargs)


def construct_rejection_strategy(noise_reducer_kwargs: Dict, additional_kwargs: Dict):
    strategy = noise_reducer_kwargs.get('strategy')

    if strategy is None:
        raise ValueError('Strategy is mandatory for constructing the instance')
    elif isinstance(strategy, RejectionStrategy):
        # This case is when the strategy already exists, e.g. saved in the model state and loaded afterwards
        return strategy
    elif isinstance(strategy, str):
        strategy_class = class_from_string(strategy)
    elif isinstance(strategy, type(RejectionStrategy)):
        strategy_class = strategy
    else:
        raise ValueError('Strategy instance can be constructed either by string representation of the full class path '
                         'or by class type')

    try:
        strategy_kwargs = noise_reducer_kwargs.pop('strategy_kwargs')

        if strategy_kwargs is None:
            strategy_kwargs = {}

        if not isinstance(strategy_kwargs, dict):
            raise ValueError('Invalid strategy kwargs type. Should be dict.')

        rejection_criteria = construct_rejection_criteria(strategy_kwargs)
        if rejection_criteria is not None:
            strategy_kwargs['rejection_criteria'] = rejection_criteria
        else:
            # This is the case that rejection_criteria exists but its None. Also remove the keyword from the dict
            if 'rejection_criteria' in strategy_kwargs:
                del strategy_kwargs['rejection_criteria']
    except KeyError:
        strategy_kwargs = {}

    init_args = get_init_args(strategy_class)

    # Convert to list to get a copy
    for arg in list(additional_kwargs.keys()):
        if arg not in init_args:
            del additional_kwargs[arg]

    strategy_kwargs.update(additional_kwargs)
    return strategy_class(**strategy_kwargs)


def construct_noise_reducer(noise_reducer_class, noise_reducer_kwargs, embedding_size, num_classes,
                            cross_batch_memory_object: CrossBatchMemory = None):

    if noise_reducer_class is None or noise_reducer_kwargs is None:
        return None

    # If we have a specified memory_size then we use this
    try:
        memory_size = noise_reducer_kwargs.pop('memory_size')
    except KeyError:
        # Else we check for existence of cross_batch_memory_object to assign this
        memory_size = 0 if cross_batch_memory_object is None else cross_batch_memory_object.memory_size

    if memory_size > 0 and cross_batch_memory_object is None:
        cross_batch_memory_object = CrossBatchMemory(loss=ContrastiveLoss(), embedding_size=embedding_size, memory_size=memory_size)
        memory_bank = MemoryBank(with_dynamic_memory=False, preallocated_object=cross_batch_memory_object)
    elif memory_size > 0 and cross_batch_memory_object is not None:
        memory_bank = MemoryBank(with_dynamic_memory=False, preallocated_object=cross_batch_memory_object)
    elif memory_size == 0 and cross_batch_memory_object is None:
        memory_bank = MemoryBank(with_dynamic_memory=True, preallocated_object=None)
    elif memory_size == 0 and cross_batch_memory_object is not None:
        warn('Both a static and a dynamic memory will be created to represent same things. '
             'Memory consumption will be higher. Be sure this is intentional.')
        memory_bank = MemoryBank(with_dynamic_memory=True, preallocated_object=None)
    else:
        assert False
        # raise ValueError(f'Invalid configuration memory_size: {memory_size}, '
        #                  f'cross_batch_memory_object: {cross_batch_memory_object}')

    noise_reducer_kwargs['memory_bank'] = memory_bank

    if noise_reducer_kwargs is None:
        noise_reducer_kwargs = {}

    if noise_reducer_class is None:
        # TODO: create passthrough reducer as with miner? Or change both
        return None

    # Some arguments need not be explictily specified multiple times, e.g. these are needed for the model and we
    # pass them as additional arguments and keep what is needed
    additional_kwargs = \
        {
            'embedding_size': embedding_size,
            'num_classes': num_classes
        }

    noise_reducer_kwargs['strategy'] = construct_rejection_strategy(noise_reducer_kwargs,
                                                                    additional_kwargs=additional_kwargs)
    return noise_reducer_class(**noise_reducer_kwargs)


def construct_reducer(reducer_class):
    return create_noisy_weighted_reducer(reducer_class)


def get_init_args(object_class):
    if object_class is None:
        return {}

    all_classes = getmro(object_class)
    init_args = set()
    for c in all_classes:
        sig = signature(c.__init__)
        for k in sig.parameters.keys():
            init_args.add(k)
    return init_args
