"""
Probability Aggregation for Ensemble Rock Avalanche Simulations

This module computes spatial probability maps from weighted ensemble
simulations, following established hazard mapping frameworks.

Outputs:
--------
1. Impact Probability Maps (P10, P50, P90)
   - Probability of flow reaching each location
   - Based on threshold exceedance across ensemble

2. Threshold Exceedance Maps
   - P(depth > threshold) at each location
   - P(velocity > threshold) at each location
   - P(pressure > threshold) at each location

3. Percentile Intensity Maps
   - 10th, 50th, 90th percentile values at each location
   - For depth, velocity, and impact pressure

Scientific Basis:
----------------
- Swiss Federal Guidelines (1997): Intensity-probability hazard zoning
- Hungr (1995): Impact pressure calculation P = 0.5 * rho * v^2
- Aaron et al. (2022): Weighted percentile methodology
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class IntensityClass(Enum):
    """
    Intensity classes based on Swiss Federal Guidelines (1997).

    These define the damage potential of mass movements based on
    the depth-velocity product (h * v in m^2/s).
    """
    HIGH = "high"       # h*v > 1 m^2/s - building destruction
    MEDIUM = "medium"   # 0.1 < h*v <= 1 m^2/s - structural damage
    LOW = "low"         # h*v <= 0.1 m^2/s - minor damage


@dataclass
class IntensityThresholds:
    """
    Configurable thresholds for intensity classification and mapping.

    Default values based on Swiss Guidelines and common practice.
    """
    # Depth thresholds (meters)
    depth_low: float = 0.5       # Below this is negligible
    depth_medium: float = 1.0    # Significant depth
    depth_high: float = 2.0      # Very deep flow

    # Velocity thresholds (m/s)
    velocity_low: float = 2.0    # Walking pace
    velocity_medium: float = 5.0  # Running pace
    velocity_high: float = 10.0   # Very fast flow

    # Pressure thresholds (kPa)
    # P = 0.5 * rho * v^2, with rho ~ 2000 kg/m^3
    pressure_low: float = 10.0    # Minor structural stress
    pressure_medium: float = 50.0  # Window failure
    pressure_high: float = 100.0   # Structural damage

    # Swiss intensity threshold (h*v product)
    hv_high: float = 1.0     # m^2/s
    hv_medium: float = 0.1   # m^2/s


@dataclass
class ProbabilityConfig:
    """Configuration for probability aggregation."""
    # Minimum value to consider as "impacted"
    min_depth_threshold: float = 0.1  # meters
    min_velocity_threshold: float = 0.5  # m/s

    # Percentiles to compute
    percentiles: List[int] = field(default_factory=lambda: [10, 50, 90])

    # Intensity thresholds
    intensity_thresholds: IntensityThresholds = field(
        default_factory=IntensityThresholds
    )

    # Density for pressure calculation (kg/m^3)
    flow_density: float = 2000.0


class ProbabilityAggregator:
    """
    Aggregate ensemble simulation results into probability maps.

    This class implements weighted percentile calculations following
    the methodology of Aaron et al. (2022), producing:
    - Impact probability maps
    - Threshold exceedance probability maps
    - Percentile intensity maps
    """

    def __init__(self, config: ProbabilityConfig = None):
        """
        Initialize the aggregator.

        Args:
            config: ProbabilityConfig with thresholds and settings
        """
        self.config = config or ProbabilityConfig()

    def compute_impact_probability(
        self,
        raster_stack: np.ndarray,
        weights: Optional[np.ndarray] = None,
        threshold: float = None
    ) -> np.ndarray:
        """
        Compute probability of impact at each cell.

        The impact probability is the weighted fraction of simulations
        where the value exceeds the threshold.

        Args:
            raster_stack: 3D array (n_sims, rows, cols) of values
            weights: 1D array of simulation weights (normalized to sum=1)
            threshold: Minimum value to consider as impact

        Returns:
            2D array of impact probabilities (0-1)
        """
        if threshold is None:
            threshold = self.config.min_depth_threshold

        n_sims = raster_stack.shape[0]

        if weights is None:
            weights = np.ones(n_sims) / n_sims

        # Binary impact mask for each simulation
        impact_mask = raster_stack > threshold

        # Weighted sum of impacts
        probability = np.zeros(raster_stack.shape[1:], dtype=np.float64)
        for i in range(n_sims):
            probability += weights[i] * impact_mask[i].astype(float)

        return probability

    def compute_threshold_exceedance(
        self,
        raster_stack: np.ndarray,
        weights: Optional[np.ndarray] = None,
        threshold: float = 1.0
    ) -> np.ndarray:
        """
        Compute probability of exceeding a specific threshold.

        P(value > threshold) at each cell, weighted by simulation weights.

        Args:
            raster_stack: 3D array (n_sims, rows, cols)
            weights: Simulation weights
            threshold: Value threshold to exceed

        Returns:
            2D array of exceedance probabilities
        """
        return self.compute_impact_probability(
            raster_stack, weights, threshold
        )

    def compute_weighted_percentile(
        self,
        raster_stack: np.ndarray,
        weights: np.ndarray,
        percentile: float
    ) -> np.ndarray:
        """
        Compute weighted percentile at each cell.

        Uses linear interpolation between sorted values, weighted by
        simulation weights.

        Args:
            raster_stack: 3D array (n_sims, rows, cols)
            weights: 1D array of weights (normalized)
            percentile: Percentile to compute (0-100)

        Returns:
            2D array of percentile values
        """
        n_sims, rows, cols = raster_stack.shape
        result = np.zeros((rows, cols), dtype=np.float64)

        # Flatten spatial dimensions for vectorized computation
        flat_stack = raster_stack.reshape(n_sims, -1)

        for j in range(flat_stack.shape[1]):
            values = flat_stack[:, j]

            # Skip if all NaN or zero
            valid_mask = np.isfinite(values) & (values > 0)
            if not np.any(valid_mask):
                continue

            valid_values = values[valid_mask]
            valid_weights = weights[valid_mask]

            # Sort by value
            sort_idx = np.argsort(valid_values)
            sorted_values = valid_values[sort_idx]
            sorted_weights = valid_weights[sort_idx]

            # Cumulative weights
            cumsum = np.cumsum(sorted_weights)
            cumsum = cumsum / cumsum[-1]  # Normalize

            # Find percentile value
            target = percentile / 100.0
            idx = np.searchsorted(cumsum, target)

            if idx == 0:
                result.flat[j] = sorted_values[0]
            elif idx >= len(sorted_values):
                result.flat[j] = sorted_values[-1]
            else:
                # Linear interpolation
                w1 = (target - cumsum[idx-1]) / (cumsum[idx] - cumsum[idx-1])
                result.flat[j] = (
                    sorted_values[idx-1] * (1 - w1) +
                    sorted_values[idx] * w1
                )

        return result

    def compute_all_percentiles(
        self,
        raster_stack: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Compute all configured percentiles.

        Args:
            raster_stack: 3D array (n_sims, rows, cols)
            weights: Simulation weights

        Returns:
            Dictionary mapping percentile to 2D array
        """
        n_sims = raster_stack.shape[0]

        if weights is None:
            weights = np.ones(n_sims) / n_sims

        results = {}
        for p in self.config.percentiles:
            results[p] = self.compute_weighted_percentile(
                raster_stack, weights, p
            )

        return results

    def compute_intensity_class_map(
        self,
        depth_stack: np.ndarray,
        velocity_stack: np.ndarray,
        weights: Optional[np.ndarray] = None,
        percentile: int = 50
    ) -> np.ndarray:
        """
        Compute Swiss-style intensity class map.

        Based on h*v product at specified percentile.

        Args:
            depth_stack: 3D array of flow depths
            velocity_stack: 3D array of velocities
            weights: Simulation weights
            percentile: Percentile to use for classification

        Returns:
            2D array of intensity classes (0=none, 1=low, 2=medium, 3=high)
        """
        # Compute h*v product for each simulation
        hv_stack = depth_stack * velocity_stack

        # Get percentile of h*v
        n_sims = hv_stack.shape[0]
        if weights is None:
            weights = np.ones(n_sims) / n_sims

        hv_percentile = self.compute_weighted_percentile(
            hv_stack, weights, percentile
        )

        # Classify
        thresholds = self.config.intensity_thresholds
        intensity_class = np.zeros_like(hv_percentile, dtype=np.int8)

        intensity_class[hv_percentile > 0] = 1  # Low
        intensity_class[hv_percentile > thresholds.hv_medium] = 2  # Medium
        intensity_class[hv_percentile > thresholds.hv_high] = 3  # High

        return intensity_class

    def compute_impact_pressure(
        self,
        velocity_stack: np.ndarray
    ) -> np.ndarray:
        """
        Compute impact pressure from velocity.

        P = 0.5 * rho * v^2

        Reference: Hungr (1995), Canadian Geotechnical Journal

        Args:
            velocity_stack: 3D array of velocities (m/s)

        Returns:
            3D array of pressures (kPa)
        """
        rho = self.config.flow_density
        # P = 0.5 * rho * v^2, convert Pa to kPa
        pressure_stack = 0.5 * rho * velocity_stack**2 / 1000.0
        return pressure_stack

    def aggregate_all_outputs(
        self,
        depth_stack: np.ndarray,
        velocity_stack: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute all standard probability outputs.

        This is the main entry point for producing a complete set of
        probability maps from ensemble simulations.

        Args:
            depth_stack: 3D array (n_sims, rows, cols) of flow depths
            velocity_stack: 3D array of velocities
            weights: Simulation weights (normalized)

        Returns:
            Dictionary with all output maps:
            - impact_probability: P(depth > threshold)
            - depth_p10, depth_p50, depth_p90: Percentile depths
            - velocity_p10, velocity_p50, velocity_p90: Percentile velocities
            - pressure_p50: Median impact pressure
            - intensity_class: Swiss-style intensity classification
            - exceed_depth_1m: P(depth > 1m)
            - exceed_velocity_5ms: P(velocity > 5 m/s)
        """
        n_sims = depth_stack.shape[0]

        if weights is None:
            weights = np.ones(n_sims) / n_sims

        outputs = {}

        # Impact probability
        outputs["impact_probability"] = self.compute_impact_probability(
            depth_stack, weights, self.config.min_depth_threshold
        )

        # Depth percentiles
        depth_percentiles = self.compute_all_percentiles(depth_stack, weights)
        for p, arr in depth_percentiles.items():
            outputs[f"depth_p{p}"] = arr

        # Velocity percentiles
        velocity_percentiles = self.compute_all_percentiles(velocity_stack, weights)
        for p, arr in velocity_percentiles.items():
            outputs[f"velocity_p{p}"] = arr

        # Pressure (from velocity)
        pressure_stack = self.compute_impact_pressure(velocity_stack)
        pressure_percentiles = self.compute_all_percentiles(pressure_stack, weights)
        for p, arr in pressure_percentiles.items():
            outputs[f"pressure_p{p}"] = arr

        # Intensity class
        outputs["intensity_class"] = self.compute_intensity_class_map(
            depth_stack, velocity_stack, weights, percentile=50
        )

        # Threshold exceedance maps
        thresholds = self.config.intensity_thresholds

        outputs["exceed_depth_low"] = self.compute_threshold_exceedance(
            depth_stack, weights, thresholds.depth_low
        )
        outputs["exceed_depth_medium"] = self.compute_threshold_exceedance(
            depth_stack, weights, thresholds.depth_medium
        )
        outputs["exceed_depth_high"] = self.compute_threshold_exceedance(
            depth_stack, weights, thresholds.depth_high
        )

        outputs["exceed_velocity_low"] = self.compute_threshold_exceedance(
            velocity_stack, weights, thresholds.velocity_low
        )
        outputs["exceed_velocity_medium"] = self.compute_threshold_exceedance(
            velocity_stack, weights, thresholds.velocity_medium
        )
        outputs["exceed_velocity_high"] = self.compute_threshold_exceedance(
            velocity_stack, weights, thresholds.velocity_high
        )

        return outputs


def compute_hazard_zones(
    impact_probability: np.ndarray,
    intensity_class: np.ndarray
) -> np.ndarray:
    """
    Compute hazard zones from probability and intensity.

    Follows Swiss-style hazard zoning matrix combining:
    - Impact probability (high/medium/low)
    - Intensity class (high/medium/low)

    Returns:
        2D array with hazard zone codes:
        - 0: No hazard
        - 1: Low hazard (blue)
        - 2: Medium hazard (yellow)
        - 3: High hazard (red)
    """
    hazard_zone = np.zeros_like(impact_probability, dtype=np.int8)

    # Probability thresholds
    p_low = 0.01   # 1% probability
    p_medium = 0.10  # 10% probability
    p_high = 0.30   # 30% probability

    # High hazard: high probability OR high intensity with medium probability
    high_mask = (
        (impact_probability > p_high) |
        ((intensity_class == 3) & (impact_probability > p_medium))
    )
    hazard_zone[high_mask] = 3

    # Medium hazard: medium probability with any intensity, or high intensity with low prob
    medium_mask = (
        ((impact_probability > p_medium) & ~high_mask) |
        ((intensity_class >= 2) & (impact_probability > p_low) & ~high_mask)
    )
    hazard_zone[medium_mask] = 2

    # Low hazard: any impact with low probability
    low_mask = (
        (impact_probability > p_low) &
        ~high_mask & ~medium_mask
    )
    hazard_zone[low_mask] = 1

    return hazard_zone
