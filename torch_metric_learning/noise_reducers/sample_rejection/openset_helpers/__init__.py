from dataclasses import dataclass, field
from typing import Union, List, Dict, Optional, Callable


@dataclass
class NearestNeighborsWeightsOptions:
    magnitude: float = 1.0
    k_neighbors: int = 20
    use_for_openset: bool = False
    use_batched_knn: bool = False


@dataclass
class DensityBasedModelsReductionOptions:
    strategy: Union[str, List] = 'coverage'
    strategy_options: Dict = field(default_factory=lambda: {'samples_coverage_ratio': 0.5})


@dataclass
class NoiseHandlingOptions:
    nn_weights: Optional[NearestNeighborsWeightsOptions] = None
    density_mr: Optional[DensityBasedModelsReductionOptions] = None
    inspection_callback: Optional[Callable] = None
