"""
Ensemble Manager for Probabilistic Rock Avalanche Simulations

This module orchestrates the complete ensemble workflow:
1. Configuration and validation
2. Parameter sample generation
3. Simulation execution tracking
4. Weight computation
5. Result aggregation
6. Report generation

The EnsembleManager maintains state throughout the workflow and provides
checkpointing for long-running ensemble jobs.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import json
import numpy as np
from datetime import datetime

from .parameter_sampler import (
    ParameterSampler,
    ParameterSample,
    SamplingConfig,
    SamplingMethod
)
from .weighting_schemes import (
    WeightingCalculator,
    WeightingMethod,
    ConfinementType,
    SimulationMetrics
)


class EnsembleStatus(Enum):
    """Status of ensemble execution."""
    CONFIGURING = "configuring"
    PREPARED = "prepared"
    RUNNING = "running"
    WEIGHTING = "weighting"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SLBLSurface:
    """
    Represents a single SLBL failure surface to include in ensemble.

    Each SLBL surface defines a potential failure volume that will be
    simulated with multiple friction parameter combinations.
    """
    slbl_id: str
    thickness_raster_path: Path
    volume_m3: float
    height_drop_m: float
    scenario_name: str
    cross_section_name: str

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "slbl_id": self.slbl_id,
            "thickness_raster_path": str(self.thickness_raster_path),
            "volume_m3": self.volume_m3,
            "height_drop_m": self.height_drop_m,
            "scenario_name": self.scenario_name,
            "cross_section_name": self.cross_section_name
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SLBLSurface':
        """Create from dictionary."""
        return cls(
            slbl_id=data["slbl_id"],
            thickness_raster_path=Path(data["thickness_raster_path"]),
            volume_m3=data["volume_m3"],
            height_drop_m=data["height_drop_m"],
            scenario_name=data["scenario_name"],
            cross_section_name=data["cross_section_name"]
        )


@dataclass
class EnsembleConfig:
    """
    Complete configuration for a probability ensemble.

    This defines all parameters needed to run the ensemble:
    - Which SLBL surfaces to include
    - How many simulations per surface
    - Parameter ranges and sampling method
    - Weighting method
    - Output options
    """
    slbl_surfaces: List[SLBLSurface]
    sims_per_slbl: int = 64
    sampling_method: SamplingMethod = SamplingMethod.GRID

    # Voellmy parameter ranges
    mu_min: float = 0.05
    mu_max: float = 0.30
    xi_min: float = 100.0
    xi_max: float = 1500.0

    # Weighting configuration
    weighting_method: WeightingMethod = WeightingMethod.EQUAL
    confinement_type: ConfinementType = ConfinementType.LATERAL

    # Simulation performance settings
    mass_per_part: float = 500000.0  # Standard default
    delta_th: float = 4.0  # Standard default
    sim_timeout: int = 7200  # Timeout per simulation in seconds (default 2 hours)

    # Output options
    percentiles: List[int] = field(default_factory=lambda: [10, 50, 90])

    # Metadata
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_simulations(self) -> int:
        """Total number of simulations in ensemble."""
        return len(self.slbl_surfaces) * self.sims_per_slbl

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "slbl_surfaces": [s.to_dict() for s in self.slbl_surfaces],
            "sims_per_slbl": self.sims_per_slbl,
            "sampling_method": self.sampling_method.value,
            "mu_min": self.mu_min,
            "mu_max": self.mu_max,
            "xi_min": self.xi_min,
            "xi_max": self.xi_max,
            "weighting_method": self.weighting_method.value,
            "confinement_type": self.confinement_type.value,
            "mass_per_part": self.mass_per_part,
            "delta_th": self.delta_th,
            "sim_timeout": self.sim_timeout,
            "percentiles": self.percentiles,
            "notes": self.notes,
            "created_at": self.created_at,
            "total_simulations": self.total_simulations
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EnsembleConfig':
        """Create from dictionary."""
        return cls(
            slbl_surfaces=[SLBLSurface.from_dict(s) for s in data["slbl_surfaces"]],
            sims_per_slbl=data["sims_per_slbl"],
            sampling_method=SamplingMethod(data["sampling_method"]),
            mu_min=data["mu_min"],
            mu_max=data["mu_max"],
            xi_min=data["xi_min"],
            xi_max=data["xi_max"],
            weighting_method=WeightingMethod(data["weighting_method"]),
            confinement_type=ConfinementType(data["confinement_type"]),
            mass_per_part=data.get("mass_per_part", 500000.0),
            delta_th=data.get("delta_th", 4.0),
            sim_timeout=data.get("sim_timeout", 7200),
            percentiles=data.get("percentiles", [10, 50, 90]),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", datetime.now().isoformat())
        )


class EnsembleManager:
    """
    Manages the complete ensemble workflow.

    This class maintains state throughout the ensemble execution,
    provides checkpointing, and coordinates all phases of the analysis.
    """

    def __init__(self, project, config: EnsembleConfig):
        """
        Initialize ensemble manager.

        Args:
            project: Project instance with directory structure
            config: EnsembleConfig defining the ensemble
        """
        self.project = project
        self.config = config
        self.status = EnsembleStatus.CONFIGURING

        # Generate unique ensemble ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ensemble_id = f"ensemble_{timestamp}"

        # Create output directory structure
        self.output_dir = project.probability_dir / self.ensemble_id
        self.ensemble_dir = self.output_dir / "simulations"
        self.results_dir = self.output_dir / "results"

        # State tracking
        self.samples: List[ParameterSample] = []
        self.simulation_results: List[Dict[str, Any]] = []
        self.metrics: List[SimulationMetrics] = []
        self.weights: Dict[int, float] = {}
        self.probability_maps: Dict[str, Path] = {}

        # SLBL lookup
        self._slbl_lookup = {s.slbl_id: s for s in config.slbl_surfaces}

    def setup_directories(self):
        """Create all output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Config and logs subdirectories
        (self.output_dir / "config").mkdir(exist_ok=True)
        (self.output_dir / "weights").mkdir(exist_ok=True)

    def save_config(self):
        """Save ensemble configuration to JSON."""
        config_path = self.output_dir / "config" / f"config_{self.ensemble_id}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def prepare_ensemble(self) -> List[ParameterSample]:
        """
        Generate all parameter samples for the ensemble.

        Creates ParameterSample objects for each SLBL surface with
        the configured number of mu/xi combinations.

        Returns:
            List of all ParameterSample objects
        """
        sampling_config = SamplingConfig(
            mu_min=self.config.mu_min,
            mu_max=self.config.mu_max,
            xi_min=self.config.xi_min,
            xi_max=self.config.xi_max,
            sampling_method=self.config.sampling_method,
            sims_per_slbl=self.config.sims_per_slbl
        )

        sampler = ParameterSampler(sampling_config)

        self.samples = []
        sample_id = 0

        for slbl in self.config.slbl_surfaces:
            surface_samples = sampler.generate_samples_for_slbl(
                slbl_id=slbl.slbl_id,
                volume_m3=slbl.volume_m3,
                height_drop_m=slbl.height_drop_m,
                base_sample_id=sample_id
            )
            self.samples.extend(surface_samples)
            sample_id += len(surface_samples)

        self.status = EnsembleStatus.PREPARED
        return self.samples

    def save_samples(self):
        """Save sample list to JSON for reference."""
        samples_path = self.output_dir / "config" / "samples.json"
        samples_data = [
            {
                "sample_id": s.sample_id,
                "mu": s.mu,
                "xi": s.xi,
                "slbl_id": s.slbl_id,
                "volume_m3": s.volume_m3,
                "height_drop_m": s.height_drop_m
            }
            for s in self.samples
        ]
        with open(samples_path, 'w') as f:
            json.dump(samples_data, f, indent=2)

    def get_slbl_for_sample(self, sample: ParameterSample) -> SLBLSurface:
        """Get the SLBL surface for a given sample."""
        return self._slbl_lookup[sample.slbl_id]

    def add_simulation_result(
        self,
        sample_id: int,
        success: bool,
        output_dir: Path = None,
        metrics: SimulationMetrics = None,
        raster_paths: Dict[str, Path] = None,
        error: str = None
    ):
        """
        Record result of a completed simulation.

        Args:
            sample_id: ID of the ParameterSample
            success: Whether simulation completed successfully
            output_dir: Path to simulation output directory
            metrics: Extracted metrics for weighting
            raster_paths: Paths to output rasters
            error: Error message if failed
        """
        result = {
            "sample_id": sample_id,
            "success": success,
            "output_dir": str(output_dir) if output_dir else None,
            "raster_paths": {k: str(v) if v else None for k, v in (raster_paths or {}).items()},
            "error": error
        }
        self.simulation_results.append(result)

        if metrics:
            self.metrics.append(metrics)

    def compute_weights(self):
        """
        Compute weights for all successful simulations.

        Uses the configured weighting method and stores results.
        """
        if not self.metrics:
            # Equal weights if no metrics
            n = len([r for r in self.simulation_results if r["success"]])
            self.weights = {
                r["sample_id"]: 1.0/n
                for r in self.simulation_results if r["success"]
            }
            return

        calculator = WeightingCalculator(
            method=self.config.weighting_method,
            confinement=self.config.confinement_type
        )

        self.weights = calculator.calculate_weights(self.metrics)

        # Save weights
        weights_path = self.output_dir / "weights" / f"weights_{self.ensemble_id}.json"
        with open(weights_path, 'w') as f:
            json.dump({
                "method": self.config.weighting_method.value,
                "confinement": self.config.confinement_type.value,
                "weights": self.weights
            }, f, indent=2)

    def get_weights_array(self) -> np.ndarray:
        """Get weights as numpy array matching simulation order."""
        successful_ids = [
            r["sample_id"] for r in self.simulation_results
            if r["success"]
        ]
        return np.array([self.weights.get(sid, 0.0) for sid in successful_ids])

    def save_report(self) -> Path:
        """
        Generate and save summary report.

        Returns:
            Path to saved report
        """
        successful = [r for r in self.simulation_results if r["success"]]

        report = {
            "ensemble_id": self.ensemble_id,
            "status": self.status.value,
            "configuration": {
                "total_simulations": len(self.samples),
                "successful": len(successful),
                "failed": len(self.simulation_results) - len(successful),
                "success_rate": len(successful) / len(self.samples) if self.samples else 0,
                "n_slbl_surfaces": len(self.config.slbl_surfaces),
                "sims_per_slbl": self.config.sims_per_slbl,
                "weighting_method": self.config.weighting_method.value,
                "confinement_type": self.config.confinement_type.value
            },
            "parameter_ranges": {
                "mu_range": [self.config.mu_min, self.config.mu_max],
                "xi_range": [self.config.xi_min, self.config.xi_max]
            },
            "output_maps": list(self.probability_maps.keys()),
            "created_at": self.config.created_at,
            "completed_at": datetime.now().isoformat()
        }

        # Add statistics from metrics
        if self.metrics:
            runouts = [m.runout_m for m in self.metrics]
            areas = [m.affected_area_m2 for m in self.metrics]
            velocities = [m.max_velocity_ms for m in self.metrics]

            report["simulation_statistics"] = {
                "runout_m": {
                    "min": min(runouts),
                    "max": max(runouts),
                    "mean": np.mean(runouts),
                    "p10": np.percentile(runouts, 10),
                    "p50": np.percentile(runouts, 50),
                    "p90": np.percentile(runouts, 90)
                },
                "affected_area_km2": {
                    "min": min(areas) / 1e6,
                    "max": max(areas) / 1e6,
                    "mean": np.mean(areas) / 1e6
                },
                "max_velocity_ms": {
                    "min": min(velocities),
                    "max": max(velocities),
                    "mean": np.mean(velocities)
                }
            }

        # Add references for reproducibility and citation
        report["references"] = [
            "Aaron et al. (2022) Landslides 19:2853-2869 - Ensemble probabilistic framework",
            "Strom et al. (2019) Landslides 16:1437-1452 - Empirical mobility constraints",
            "Brideau et al. (2021) WLF5 pp.321-327 - H/L exceedance relationships",
            "Hungr (1995) Can. Geotech. J. 32:610-623 - Voellmy model formulation"
        ]

        report_path = self.results_dir / "summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report_path


def create_ensemble_from_slbl_batch(
    project,
    slbl_results: List[Dict],
    sims_per_slbl: int = 64,
    weighting_method: WeightingMethod = WeightingMethod.EQUAL
) -> EnsembleConfig:
    """
    Create ensemble configuration from SLBL batch results.

    Convenience function to convert SLBL analysis outputs to
    an ensemble configuration.

    Args:
        project: Project instance
        slbl_results: List of SLBL result dictionaries
        sims_per_slbl: Simulations per SLBL surface
        weighting_method: Weighting method to use

    Returns:
        EnsembleConfig ready for execution
    """
    surfaces = []
    for result in slbl_results:
        surfaces.append(SLBLSurface(
            slbl_id=result.get("slbl_id", result.get("label", "unknown")),
            thickness_raster_path=Path(result["thickness_raster_path"]),
            volume_m3=result.get("volume_m3", 0),
            height_drop_m=result.get("height_drop_m", 0),
            scenario_name=result.get("scenario_name", ""),
            cross_section_name=result.get("cross_section_name", "")
        ))

    return EnsembleConfig(
        slbl_surfaces=surfaces,
        sims_per_slbl=sims_per_slbl,
        weighting_method=weighting_method
    )
