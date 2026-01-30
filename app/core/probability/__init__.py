"""
Probability Core Module for Rock Avalanche Ensemble Analysis

This package provides scientifically-grounded probabilistic runout mapping
from ensemble AvaFrame simulations. It generates P10/P50/P90 probability
maps for multiple hazard metrics suitable for emergency response.

Scientific Foundation:
---------------------
- Aaron et al. (2022): Bayesian ensemble framework, Landslides 19:2853-2869
- Strom et al. (2019): Area-energy relationships, Landslides 16:1437-1452
- Brideau et al. (2021): H/L exceedance probability, WLF5 Proceedings
- Swiss Guidelines (1997): Intensity-probability hazard zoning

Modules:
--------
- parameter_sampler: Generate mu/xi parameter combinations
- weighting_schemes: Equal and empirically-calibrated weighting
- probability_aggregator: Compute spatial probability maps
- ensemble_manager: Orchestrate ensemble workflow
- probability_handler: Job queue integration
"""

from .parameter_sampler import (
    ParameterSampler,
    ParameterSample,
    SamplingConfig,
    SamplingMethod,
    get_recommended_ranges_for_volume,
    VOELLMY_REFERENCE_VALUES
)

from .weighting_schemes import (
    WeightingCalculator,
    WeightingMethod,
    ConfinementType,
    SimulationMetrics,
    validate_simulation_against_empirical,
    get_weighting_method_description,
    STROM_ATOTAL_VHMAX_COEFFICIENTS,
    STROM_RUNOUT_COEFFICIENTS,
    BRIDEAU_HL_COEFFICIENTS
)

from .probability_aggregator import (
    ProbabilityAggregator,
    ProbabilityConfig,
    IntensityThresholds,
    IntensityClass,
    compute_hazard_zones
)

from .ensemble_manager import (
    EnsembleManager,
    EnsembleConfig,
    EnsembleStatus,
    SLBLSurface,
    create_ensemble_from_slbl_batch
)

from .probability_handler import (
    run_probability_ensemble,
    extract_metrics_from_result,
    aggregate_ensemble_results,
    load_ensemble_results,
    compare_weighting_methods
)

__version__ = "1.0.0"
__author__ = "Gustav / MTLab"

__all__ = [
    # Parameter sampling
    "ParameterSampler",
    "ParameterSample",
    "SamplingConfig",
    "SamplingMethod",
    "get_recommended_ranges_for_volume",
    "VOELLMY_REFERENCE_VALUES",

    # Weighting
    "WeightingCalculator",
    "WeightingMethod",
    "ConfinementType",
    "SimulationMetrics",
    "validate_simulation_against_empirical",
    "get_weighting_method_description",
    "STROM_ATOTAL_VHMAX_COEFFICIENTS",
    "STROM_RUNOUT_COEFFICIENTS",
    "BRIDEAU_HL_COEFFICIENTS",

    # Probability aggregation
    "ProbabilityAggregator",
    "ProbabilityConfig",
    "IntensityThresholds",
    "IntensityClass",
    "compute_hazard_zones",

    # Ensemble management
    "EnsembleManager",
    "EnsembleConfig",
    "EnsembleStatus",
    "SLBLSurface",
    "create_ensemble_from_slbl_batch",

    # Handler functions
    "run_probability_ensemble",
    "extract_metrics_from_result",
    "aggregate_ensemble_results",
    "load_ensemble_results",
    "compare_weighting_methods"
]
