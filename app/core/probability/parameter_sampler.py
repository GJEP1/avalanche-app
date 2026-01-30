"""
Parameter Sampling for Probabilistic Rock Avalanche Simulations

This module provides parameter sampling strategies for generating ensemble
simulations with varying Voellmy-Salm friction parameters (mu, xi).

Scientific References:
----------------------
- Aaron, J., McDougall, S., Mitchell, A., Korup, O., & Nolde, N. (2022).
  Probabilistic prediction of rock avalanche runout using a numerical model.
  Landslides, 19, 2853-2869. DOI: 10.1007/s10346-022-01939-y

  Parameter ranges from 31 global rock avalanche back-analyses:
  - mu (friction coefficient): 0.05 to 0.25
  - xi (turbulence coefficient): 100 to 2000 m/s^2

- Quan Luna, B., Cepeda, J., Remaitre, A., & van Asch, T. (2012).
  Application of a Monte Carlo method for modeling debris flow run-out.
  Poster presentation, EGU General Assembly.

  Used grid-based sampling with 5000 Monte Carlo realizations,
  demonstrating the value of systematic parameter space coverage.

- Fischer, J.T., et al. (2020). r.avaflow parameter sensitivity analysis.
  Geoscientific Model Development.

  Voellmy parameter ranges for various mass movement types.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import numpy as np


class SamplingMethod(Enum):
    """Available parameter sampling methods."""
    GRID = "grid"
    LATIN_HYPERCUBE = "latin_hypercube"
    RANDOM = "random"


@dataclass
class ParameterSample:
    """
    Single parameter combination for simulation.

    Attributes:
        mu: Coulomb friction coefficient (dimensionless, 0.05-0.30)
        xi: Turbulence coefficient (m/s^2, 100-1500)
        slbl_id: Reference to SLBL thickness raster
        volume_m3: Estimated failure volume in cubic meters
        height_drop_m: Height drop from crown to deposit toe (m)
        sample_id: Unique identifier within ensemble
    """
    mu: float
    xi: float
    slbl_id: str
    volume_m3: float
    height_drop_m: float
    sample_id: int

    def __post_init__(self):
        """Validate parameter ranges."""
        if not 0.01 <= self.mu <= 0.60:
            raise ValueError(f"mu must be between 0.01 and 0.60, got {self.mu}")
        if not 50 <= self.xi <= 3000:
            raise ValueError(f"xi must be between 50 and 3000 m/s^2, got {self.xi}")

    @property
    def effective_friction_angle_deg(self) -> float:
        """
        Convert mu to equivalent friction angle in degrees.

        tan(phi) = mu -> phi = arctan(mu)
        """
        return np.degrees(np.arctan(self.mu))

    @property
    def estimated_max_velocity_ms(self) -> float:
        """
        Rough order-of-magnitude estimate of maximum velocity.

        WARNING: This is an approximate estimator for preview purposes only.
        Do not use for hazard classification or design. Actual velocities
        should be taken from simulation outputs.

        Based on simplified Voellmy energy balance:
        v_max ≈ sqrt(ξ × H × sin(θ)), assuming θ ≈ 30°

        Reference: Conceptually follows Hungr (1995), Can. Geotech. J. 32:610-623
        """
        sin_theta = 0.5  # sin(30 deg)
        return np.sqrt(self.xi * self.height_drop_m * sin_theta)


@dataclass
class SamplingConfig:
    """
    Configuration for parameter sampling.

    Default ranges based on Aaron et al. (2022) analysis of 31 rock avalanches:
    - All 31 cases yielded best-fit mu < 0.25
    - Large-volume events (>10^7 m^3) often required mu < 0.10
    - xi showed wide range depending on flow conditions

    Attributes:
        mu_min: Minimum friction coefficient (default 0.05)
        mu_max: Maximum friction coefficient (default 0.30)
        xi_min: Minimum turbulence coefficient in m/s^2 (default 100)
        xi_max: Maximum turbulence coefficient in m/s^2 (default 1500)
        sampling_method: Strategy for sampling parameter space
        sims_per_slbl: Number of simulations per SLBL surface
    """
    mu_min: float = 0.05
    mu_max: float = 0.30
    xi_min: float = 100.0
    xi_max: float = 1500.0
    sampling_method: SamplingMethod = SamplingMethod.GRID
    sims_per_slbl: int = 64

    def __post_init__(self):
        """Validate configuration."""
        if self.mu_min >= self.mu_max:
            raise ValueError("mu_min must be less than mu_max")
        if self.xi_min >= self.xi_max:
            raise ValueError("xi_min must be less than xi_max")
        if self.sims_per_slbl not in [32, 64, 128]:
            raise ValueError("sims_per_slbl must be 32, 64, or 128")

    @property
    def mu_range(self) -> Tuple[float, float]:
        """Return mu range as tuple."""
        return (self.mu_min, self.mu_max)

    @property
    def xi_range(self) -> Tuple[float, float]:
        """Return xi range as tuple."""
        return (self.xi_min, self.xi_max)


class ParameterSampler:
    """
    Generate parameter samples for ensemble simulations.

    This class implements multiple sampling strategies for exploring the
    Voellmy-Salm parameter space (mu, xi) used in rock avalanche modeling.

    Scientific Basis:
    -----------------
    The Voellmy-Salm model describes basal resistance as:

        tau_b = mu * sigma_n + (rho*g / xi) * v^2

    Where:
    - tau_b = basal shear stress
    - mu = Coulomb friction coefficient (dominates at low velocity)
    - sigma_n = normal stress
    - xi = turbulence coefficient (dominates at high velocity)
    - v = flow velocity

    Parameter Sensitivity:
    ---------------------
    From Hergarten (2024), Earth Surface Dynamics:
    - mu primarily controls runout distance
    - xi primarily controls flow velocity
    - Effective friction decreases with flow thickness:
      mu_eff = mu + (rho*g/xi) * (v^2/h)

    This volume-dependent effective friction explains the observed
    decrease in H/L ratio (Heim's ratio) with increasing volume.

    References:
    -----------
    - Aaron et al. (2022): mu = 0.05-0.25 from 31 rock avalanche back-analyses
    - Hergarten (2024): Modified Voellmy rheology and volume-mobility scaling
    - McKay et al. (1979): Latin Hypercube Sampling methodology
    """

    def __init__(self, config: SamplingConfig):
        """
        Initialize sampler with configuration.

        Args:
            config: SamplingConfig with parameter ranges and method
        """
        self.config = config

    def generate_samples(self, n_samples: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Generate parameter samples using configured method.

        Args:
            n_samples: Number of samples (default: config.sims_per_slbl)

        Returns:
            List of (mu, xi) tuples
        """
        n = n_samples or self.config.sims_per_slbl

        if self.config.sampling_method == SamplingMethod.GRID:
            return self.generate_grid_samples(n)
        elif self.config.sampling_method == SamplingMethod.LATIN_HYPERCUBE:
            return self.generate_latin_hypercube_samples(n)
        elif self.config.sampling_method == SamplingMethod.RANDOM:
            return self.generate_random_samples(n)
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")

    def generate_grid_samples(self, n_samples: int) -> List[Tuple[float, float]]:
        """
        Generate regular grid of mu-xi combinations.

        Grid sampling ensures coverage of parameter space edges, which is
        important for capturing extreme runout scenarios (low mu, high xi)
        and conservative scenarios (high mu, low xi).

        The grid is designed with slightly more resolution in mu because
        runout distance is more sensitive to friction coefficient than
        to turbulence (Hergarten, 2024).

        Args:
            n_samples: Target number of samples (actual may differ slightly)

        Returns:
            List of (mu, xi) tuples covering parameter grid

        Reference:
            Quan Luna et al. (2012) used grid-based initial distribution
            for Monte Carlo debris flow simulations.
        """
        # Calculate grid dimensions
        # Use 60/40 split favoring mu resolution
        n_mu = int(np.ceil(np.sqrt(n_samples * 0.6)))
        n_xi = int(np.ceil(n_samples / n_mu))

        # Generate evenly spaced values
        mu_values = np.linspace(
            self.config.mu_min,
            self.config.mu_max,
            n_mu
        )
        xi_values = np.linspace(
            self.config.xi_min,
            self.config.xi_max,
            n_xi
        )

        # Create grid
        samples = []
        for mu in mu_values:
            for xi in xi_values:
                samples.append((float(mu), float(xi)))

        # Trim to exact count if needed
        return samples[:n_samples]

    def generate_latin_hypercube_samples(self, n_samples: int) -> List[Tuple[float, float]]:
        """
        Latin Hypercube Sampling for improved space coverage.

        LHS ensures each parameter range is sampled uniformly while
        minimizing clustering that can occur with random sampling.
        This provides better coverage of the parameter space with
        fewer samples than pure random sampling.

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of (mu, xi) tuples with LHS distribution

        Reference:
            McKay, M.D., Beckman, R.J., & Conover, W.J. (1979).
            A comparison of three methods for selecting values of
            input variables in the analysis of output from a computer code.
            Technometrics, 21(2), 239-245.
        """
        try:
            from scipy.stats import qmc

            # Create LHS sampler for 2D space
            sampler = qmc.LatinHypercube(d=2, seed=42)
            samples_unit = sampler.random(n=n_samples)

        except ImportError:
            # Fallback to simple stratified random sampling
            samples_unit = self._simple_lhs(n_samples)

        # Scale to parameter ranges
        samples = []
        for s in samples_unit:
            mu = self.config.mu_min + s[0] * (self.config.mu_max - self.config.mu_min)
            xi = self.config.xi_min + s[1] * (self.config.xi_max - self.config.xi_min)
            samples.append((float(mu), float(xi)))

        return samples

    def _simple_lhs(self, n_samples: int) -> np.ndarray:
        """
        Simple LHS implementation without scipy.

        Divides each dimension into n equal strata and samples
        one point from each stratum with random permutation.
        """
        np.random.seed(42)

        # Create stratified samples
        samples = np.zeros((n_samples, 2))

        for dim in range(2):
            # Random point within each stratum
            base = np.arange(n_samples) / n_samples
            samples[:, dim] = base + np.random.random(n_samples) / n_samples
            # Shuffle
            np.random.shuffle(samples[:, dim])

        return samples

    def generate_random_samples(self, n_samples: int) -> List[Tuple[float, float]]:
        """
        Generate purely random samples within parameter ranges.

        Less efficient than LHS but simpler and provides baseline
        for comparison.

        Args:
            n_samples: Number of random samples

        Returns:
            List of (mu, xi) tuples with uniform random distribution
        """
        np.random.seed(42)

        mu_samples = np.random.uniform(
            self.config.mu_min,
            self.config.mu_max,
            n_samples
        )
        xi_samples = np.random.uniform(
            self.config.xi_min,
            self.config.xi_max,
            n_samples
        )

        return [(float(mu), float(xi)) for mu, xi in zip(mu_samples, xi_samples)]

    def generate_samples_for_slbl(
        self,
        slbl_id: str,
        volume_m3: float,
        height_drop_m: float,
        base_sample_id: int = 0
    ) -> List[ParameterSample]:
        """
        Generate parameter samples for a single SLBL surface.

        Creates complete ParameterSample objects with metadata
        linking each sample to its source SLBL result.

        Args:
            slbl_id: Identifier for SLBL thickness raster
            volume_m3: Estimated failure volume in m^3
            height_drop_m: Height from crown to deposit toe (m)
            base_sample_id: Starting ID for numbering samples

        Returns:
            List of ParameterSample objects ready for simulation
        """
        param_pairs = self.generate_samples()

        samples = []
        for i, (mu, xi) in enumerate(param_pairs):
            samples.append(ParameterSample(
                mu=mu,
                xi=xi,
                slbl_id=slbl_id,
                volume_m3=volume_m3,
                height_drop_m=height_drop_m,
                sample_id=base_sample_id + i
            ))

        return samples

    def get_parameter_grid_info(self) -> dict:
        """
        Get information about the parameter grid for documentation.

        Returns:
            Dictionary with grid statistics and coverage information
        """
        samples = self.generate_samples()
        mu_values = [s[0] for s in samples]
        xi_values = [s[1] for s in samples]

        return {
            "n_samples": len(samples),
            "sampling_method": self.config.sampling_method.value,
            "mu_range": self.config.mu_range,
            "xi_range": self.config.xi_range,
            "mu_values": {
                "min": min(mu_values),
                "max": max(mu_values),
                "n_unique": len(set(np.round(mu_values, 4)))
            },
            "xi_values": {
                "min": min(xi_values),
                "max": max(xi_values),
                "n_unique": len(set(np.round(xi_values, 0)))
            }
        }


def get_recommended_ranges_for_volume(volume_m3: float) -> SamplingConfig:
    """
    Get recommended parameter ranges based on failure volume.

    Larger rock avalanches typically exhibit lower effective friction
    (enhanced mobility), so the parameter ranges can be adjusted
    based on volume to focus computational effort.

    Based on volume-mobility relationships from:
    - Strom et al. (2019): L vs V relationships by confinement
    - Aaron et al. (2022): All 31 cases had mu < 0.25
    - Scheidegger (1973): Original H/L vs V relationship

    Args:
        volume_m3: Estimated failure volume

    Returns:
        SamplingConfig with volume-appropriate parameter ranges
    """
    if volume_m3 >= 1e8:  # >100 Mm^3 - very large
        # Very large events show enhanced mobility
        return SamplingConfig(
            mu_min=0.03,
            mu_max=0.15,
            xi_min=300,
            xi_max=1500
        )
    elif volume_m3 >= 1e7:  # 10-100 Mm^3 - large
        return SamplingConfig(
            mu_min=0.05,
            mu_max=0.20,
            xi_min=200,
            xi_max=1200
        )
    elif volume_m3 >= 1e6:  # 1-10 Mm^3 - moderate
        return SamplingConfig(
            mu_min=0.08,
            mu_max=0.25,
            xi_min=150,
            xi_max=1000
        )
    else:  # <1 Mm^3 - small
        # Smaller failures behave more like frictional slides
        return SamplingConfig(
            mu_min=0.15,
            mu_max=0.35,
            xi_min=100,
            xi_max=800
        )


# Typical Voellmy parameter ranges inferred from literature and modeling practice
# Note: These are practical modeling ranges, not strict values from single papers.
# They synthesize back-analysis results, sensitivity studies, and runout code conventions.
VOELLMY_REFERENCE_VALUES = {
    "rock_avalanche_dry": {
        "mu": {"min": 0.05, "max": 0.30, "typical": 0.15},
        "xi": {"min": 100, "max": 2000, "typical": 500},
        "basis": "Back-analyses and sensitivity ranges (Aaron et al. 2022; McDougall & Hungr 2004)"
    },
    "rock_avalanche_on_glacier": {
        "mu": {"min": 0.03, "max": 0.15, "typical": 0.08},
        "xi": {"min": 500, "max": 2000, "typical": 1000},
        "basis": "Low effective friction on ice; values from post-1996 modeling practice (RAMMS, DAN3D)"
    },
    "debris_flow": {
        "mu": {"min": 0.07, "max": 0.30, "typical": 0.15},
        "xi": {"min": 100, "max": 800, "typical": 400},
        "basis": "Typical Voellmy-type debris-flow modeling ranges (Rickenmann 1999; RAMMS documentation)"
    },
    "snow_avalanche_dense": {
        "mu": {"min": 0.15, "max": 0.35, "typical": 0.25},
        "xi": {"min": 1000, "max": 2000, "typical": 1500},
        "basis": "Typical RAMMS dense-flow calibration ranges (Bartelt et al. 2012-2017)"
    }
}
