"""
Simulation Weighting Schemes for Probabilistic Rock Avalanche Analysis

This module implements various weighting strategies for ensemble simulations
based on empirical landslide relationships from published literature.

Scientific Foundation:
---------------------
The fundamental premise is that simulations producing results consistent
with observed rock avalanche behavior should receive higher weights in
probability calculations. This is implemented through comparison with
established empirical relationships.

Key References:
--------------
1. Strom, Li & Lan (2019) - Landslides 16:1437-1452
   "Rock avalanche mobility: optimal characterization and the effects of confinement"
   - N = 595 Central Asian rock avalanches
   - Area vs V*H relationships by confinement type
   - R^2 > 0.92 for all confinement categories

2. Brideau et al. (2021) - WLF5 Proceedings
   "Empirical Relationships to Estimate the Probability of Runout Exceedance"
   - H/L vs Volume probability of exceedance
   - N = 288 rock avalanche cases

3. Aaron et al. (2022) - Landslides 19:2853-2869
   "Probabilistic prediction of rock avalanche runout using a numerical model"
   - Bayesian-inspired framework for combining ensemble simulation results
   - Empirical relationships used as probabilistic constraints
   - Distinguished calibration vs mechanistic uncertainty
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class WeightingMethod(Enum):
    """Available simulation weighting methods."""
    EQUAL = "equal"
    EMPIRICAL_STROM = "empirical_strom"
    EMPIRICAL_BRIDEAU = "empirical_brideau"
    COMBINED = "combined"


class ConfinementType(Enum):
    """
    Confinement categories from Strom et al. (2019).

    The confinement type significantly affects mobility:
    - Lateral: Channelized flow, longest runout
    - Frontal: Flow blocked by opposing slope
    - Unconfined: Spreading on open slope
    """
    LATERAL = "lateral"
    FRONTAL = "frontal"
    UNCONFINED = "unconfined"


# Strom et al. (2019) regression coefficients for Area vs V*H
# Table 1, rightmost section: A_total = f(V × H_max)
# Equation: log(A_total) = a + b * log(V * H_max)
# Units: V in 10^6 m^3, H_max in km, A_total in km^2
STROM_ATOTAL_VHMAX_COEFFICIENTS = {
    ConfinementType.FRONTAL: {
        "a": 0.9791,
        "a_se": 0.0168,  # Standard error
        "b": 0.4849,
        "b_se": 0.0076,  # Standard error
        "R2": 0.9258,
        "N": 294,
        "description": "Frontally confined (blocked by slope)"
    },
    ConfinementType.LATERAL: {
        "a": 1.0884,
        "a_se": 0.0357,  # Standard error
        "b": 0.5497,
        "b_se": 0.0163,  # Standard error
        "R2": 0.9267,
        "N": 68,
        "description": "Laterally confined (valley/channel)"
    },
    ConfinementType.UNCONFINED: {
        "a": 1.2537,
        "a_se": 0.0497,  # Standard error
        "b": 0.5668,
        "b_se": 0.0172,  # Standard error
        "R2": 0.9361,
        "N": 71,
        "description": "Unconfined (open slope/fan)"
    }
}

# Strom et al. (2019) L vs V*H relationships
# Table 3: log(L) = a + b * log(V * H_max)
STROM_RUNOUT_COEFFICIENTS = {
    ConfinementType.LATERAL: {
        "a": 0.5621,
        "b": 0.4012,
        "R2": 0.8856
    },
    ConfinementType.FRONTAL: {
        "a": 0.4587,
        "b": 0.3845,
        "R2": 0.8521
    },
    ConfinementType.UNCONFINED: {
        "a": 0.4892,
        "b": 0.4156,
        "R2": 0.8978
    }
}

# Brideau et al. (2021) H/L vs Volume relationship for rock avalanches
# log(H/L) = a + b * log(V)
BRIDEAU_HL_COEFFICIENTS = {
    "rock_avalanche": {
        "a": 0.469,
        "b": -0.137,
        "sigma": 0.18,  # Standard deviation of residuals
        "N": 288,
        "V_units": "m^3",  # Volume in cubic meters
        "description": "Rock avalanche H/L mobility"
    }
}


@dataclass
class SimulationMetrics:
    """
    Metrics extracted from a completed simulation for weighting.

    These metrics are compared against empirical relationships to
    assign weights reflecting physical plausibility.
    """
    sample_id: int
    slbl_id: str
    mu: float
    xi: float
    volume_m3: float
    height_drop_m: float
    runout_m: float
    affected_area_m2: float
    max_velocity_ms: float
    max_depth_m: float

    @property
    def volume_Mm3(self) -> float:
        """Volume in millions of cubic meters."""
        return self.volume_m3 / 1e6

    @property
    def height_km(self) -> float:
        """Height drop in kilometers."""
        return self.height_drop_m / 1000

    @property
    def VH_product(self) -> float:
        """V * H product in Mm^3 * km (Strom mobility parameter)."""
        return self.volume_Mm3 * self.height_km

    @property
    def area_km2(self) -> float:
        """Affected area in square kilometers."""
        return self.affected_area_m2 / 1e6

    @property
    def runout_km(self) -> float:
        """Runout distance in kilometers."""
        return self.runout_m / 1000

    @property
    def HL_ratio(self) -> float:
        """Height-to-Length ratio (Heim's ratio)."""
        if self.runout_m > 0:
            return self.height_drop_m / self.runout_m
        return float('inf')


class WeightingCalculator:
    """
    Calculate simulation weights based on empirical relationships.

    The weighting approach follows Aaron et al. (2022):
    - Simulations consistent with observations get higher weights
    - Weights are normalized to sum to 1.0
    - Multiple empirical relationships can be combined
    """

    def __init__(
        self,
        method: WeightingMethod = WeightingMethod.EQUAL,
        confinement: ConfinementType = ConfinementType.LATERAL
    ):
        """
        Initialize the weighting calculator.

        Args:
            method: Weighting method to use
            confinement: Confinement type for Strom relationships
        """
        self.method = method
        self.confinement = confinement

    def calculate_weights(
        self,
        metrics_list: List[SimulationMetrics]
    ) -> Dict[int, float]:
        """
        Calculate normalized weights for all simulations.

        Args:
            metrics_list: List of SimulationMetrics from completed simulations

        Returns:
            Dictionary mapping sample_id to normalized weight (sums to 1.0)
        """
        if not metrics_list:
            return {}

        if self.method == WeightingMethod.EQUAL:
            # All simulations weighted equally
            n = len(metrics_list)
            return {m.sample_id: 1.0 / n for m in metrics_list}

        elif self.method == WeightingMethod.EMPIRICAL_STROM:
            return self._calculate_strom_weights(metrics_list)

        elif self.method == WeightingMethod.EMPIRICAL_BRIDEAU:
            return self._calculate_brideau_weights(metrics_list)

        elif self.method == WeightingMethod.COMBINED:
            return self._calculate_combined_weights(metrics_list)

        else:
            raise ValueError(f"Unknown weighting method: {self.method}")

    def _calculate_strom_weights(
        self,
        metrics_list: List[SimulationMetrics]
    ) -> Dict[int, float]:
        """
        Calculate weights using Strom et al. (2019) Area vs V*H relationship.

        Simulations producing areas close to empirical prediction get
        higher weights. Uses log-normal likelihood based on regression
        scatter.
        """
        coef = STROM_ATOTAL_VHMAX_COEFFICIENTS[self.confinement]

        weights = {}
        for m in metrics_list:
            if m.VH_product > 0 and m.area_km2 > 0:
                # Predicted log(Area) from empirical relationship
                log_VH = np.log10(m.VH_product)
                log_A_predicted = coef["a"] + coef["b"] * log_VH

                # Observed log(Area)
                log_A_observed = np.log10(m.area_km2)

                # Residual (difference from prediction)
                residual = abs(log_A_observed - log_A_predicted)

                # Weight based on residual (exponential decay)
                # Typical scatter is ~0.3 log units
                sigma = 0.3
                weight = np.exp(-0.5 * (residual / sigma) ** 2)
            else:
                weight = 0.001  # Very small weight for invalid cases

            weights[m.sample_id] = weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _calculate_brideau_weights(
        self,
        metrics_list: List[SimulationMetrics]
    ) -> Dict[int, float]:
        """
        Calculate weights using Brideau et al. (2021) H/L vs Volume relationship.

        Simulations with H/L ratios consistent with empirical data
        receive higher weights.
        """
        coef = BRIDEAU_HL_COEFFICIENTS["rock_avalanche"]

        weights = {}
        for m in metrics_list:
            if m.volume_m3 > 0 and m.HL_ratio > 0 and m.HL_ratio < float('inf'):
                # Predicted log(H/L) from empirical relationship
                log_V = np.log10(m.volume_m3)
                log_HL_predicted = coef["a"] + coef["b"] * log_V

                # Observed log(H/L)
                log_HL_observed = np.log10(m.HL_ratio)

                # Residual
                residual = abs(log_HL_observed - log_HL_predicted)

                # Weight based on residual
                sigma = coef["sigma"]
                weight = np.exp(-0.5 * (residual / sigma) ** 2)
            else:
                weight = 0.001

            weights[m.sample_id] = weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _calculate_combined_weights(
        self,
        metrics_list: List[SimulationMetrics]
    ) -> Dict[int, float]:
        """
        Combine Strom and Brideau weights using geometric mean.

        This balances both area and mobility constraints.
        """
        strom_weights = self._calculate_strom_weights(metrics_list)
        brideau_weights = self._calculate_brideau_weights(metrics_list)

        combined = {}
        for m in metrics_list:
            w1 = strom_weights.get(m.sample_id, 0.001)
            w2 = brideau_weights.get(m.sample_id, 0.001)
            combined[m.sample_id] = np.sqrt(w1 * w2)

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        return combined

    def get_empirical_prediction(
        self,
        volume_m3: float,
        height_drop_m: float
    ) -> Dict[str, float]:
        """
        Get empirical predictions for given volume and height drop.

        Useful for comparing simulation results to expected values.

        Returns:
            Dictionary with predicted area, runout, and H/L ratio
        """
        V_Mm3 = volume_m3 / 1e6
        H_km = height_drop_m / 1000
        VH = V_Mm3 * H_km

        # Strom area prediction
        area_coef = STROM_ATOTAL_VHMAX_COEFFICIENTS[self.confinement]
        if VH > 0:
            log_A = area_coef["a"] + area_coef["b"] * np.log10(VH)
            area_km2 = 10 ** log_A
        else:
            area_km2 = 0

        # Strom runout prediction
        runout_coef = STROM_RUNOUT_COEFFICIENTS[self.confinement]
        if VH > 0:
            log_L = runout_coef["a"] + runout_coef["b"] * np.log10(VH)
            runout_km = 10 ** log_L
        else:
            runout_km = 0

        # Brideau H/L prediction
        hl_coef = BRIDEAU_HL_COEFFICIENTS["rock_avalanche"]
        if volume_m3 > 0:
            log_HL = hl_coef["a"] + hl_coef["b"] * np.log10(volume_m3)
            HL_ratio = 10 ** log_HL
        else:
            HL_ratio = 0

        return {
            "predicted_area_km2": area_km2,
            "predicted_runout_km": runout_km,
            "predicted_HL_ratio": HL_ratio,
            "VH_product": VH
        }


def validate_simulation_against_empirical(
    metrics: SimulationMetrics,
    confinement: ConfinementType = ConfinementType.LATERAL
) -> Dict[str, any]:
    """
    Validate a single simulation against empirical relationships.

    Returns comparison of simulated vs predicted values with
    quality scores.

    Args:
        metrics: Simulation metrics to validate
        confinement: Confinement type for Strom relationships

    Returns:
        Dictionary with predictions, observations, and residuals
    """
    calculator = WeightingCalculator(confinement=confinement)
    predictions = calculator.get_empirical_prediction(
        metrics.volume_m3,
        metrics.height_drop_m
    )

    # Calculate residuals
    area_residual = None
    if predictions["predicted_area_km2"] > 0 and metrics.area_km2 > 0:
        area_residual = (
            np.log10(metrics.area_km2) -
            np.log10(predictions["predicted_area_km2"])
        )

    runout_residual = None
    if predictions["predicted_runout_km"] > 0 and metrics.runout_km > 0:
        runout_residual = (
            np.log10(metrics.runout_km) -
            np.log10(predictions["predicted_runout_km"])
        )

    hl_residual = None
    if predictions["predicted_HL_ratio"] > 0 and metrics.HL_ratio > 0:
        hl_residual = (
            np.log10(metrics.HL_ratio) -
            np.log10(predictions["predicted_HL_ratio"])
        )

    return {
        "predictions": predictions,
        "observations": {
            "area_km2": metrics.area_km2,
            "runout_km": metrics.runout_km,
            "HL_ratio": metrics.HL_ratio
        },
        "residuals": {
            "area_log_residual": area_residual,
            "runout_log_residual": runout_residual,
            "HL_log_residual": hl_residual
        },
        "quality_flags": {
            "area_within_1sigma": abs(area_residual or 999) < 0.3,
            "runout_within_1sigma": abs(runout_residual or 999) < 0.3,
            "HL_within_1sigma": abs(hl_residual or 999) < 0.18
        }
    }


def get_weighting_method_description(method: WeightingMethod) -> str:
    """Get human-readable description of a weighting method."""
    descriptions = {
        WeightingMethod.EQUAL: (
            "Equal weights: All simulations contribute equally to probability "
            "estimates. This is the most conservative approach, making no "
            "assumptions about which parameter combinations are more likely."
        ),
        WeightingMethod.EMPIRICAL_STROM: (
            "Strom et al. (2019) weighting: Simulations are weighted using "
            "empirical constraints from the Area vs V×H relationship derived "
            "from 595 Central Asian rock avalanches. Simulations producing "
            "areas consistent with observations receive higher weights."
        ),
        WeightingMethod.EMPIRICAL_BRIDEAU: (
            "Brideau et al. (2021) weighting: Simulations are weighted using "
            "empirical constraints from the H/L vs Volume relationship derived "
            "from 288 rock avalanche cases. Simulations with mobility consistent "
            "with observations receive higher weights."
        ),
        WeightingMethod.COMBINED: (
            "Combined weighting (geometric mean of Strom and Brideau). "
            "A pragmatic combination of independent empirical constraints that "
            "penalizes simulations deviating from either mobility relationship. "
            "Conceptually similar to Bayesian model averaging."
        )
    }
    return descriptions.get(method, "Unknown weighting method")
