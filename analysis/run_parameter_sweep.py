"""
AVAFRAME Parameter Sweep Runner
================================
Runs multiple AVAFRAME simulations across a grid of Voellmy parameters (μ, ξ).
Results are organized for use with the Runout Explorer visualization tool.

Usage:
    python run_parameter_sweep.py --dem /path/to/dem.asc --rel /path/to/release.asc \
        --mu-min 0.025 --mu-max 0.4 --mu-steps 16 \
        --xi-min 250 --xi-max 2000 --xi-steps 8 \
        --output /path/to/output

Author: Gustav / MTLab
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import numpy as np
import configparser
from datetime import datetime
import json


def create_avaframe_config(mu: float, xi: float, 
                           cellsize: float = 10.0,
                           rho: float = 2500.0) -> str:
    """Generate AVAFRAME configuration for rock avalanche with given parameters."""
    config_content = f"""# AVAFRAME Rock Avalanche Configuration
# Generated: {datetime.now().isoformat()}
# Voellmy Parameters: mu={mu:.4f}, xi={xi:.1f}

[com1DFA_com1DFA_override]
# Mesh settings
meshCellSize = {cellsize}

# Material properties
rho = {rho}

# Friction model: Voellmy
frictModel = Voellmy
muvoellmy = {mu}
xsivoellmy = {xi}

# SPH settings (recommended for rock avalanche)
sphOption = 3
massPerPart = 280000
deltaTh = 1.0

# Particle management
splitOption = 1

# Output settings
resType = ppr|pft|pfv|FT
tSteps = 0:1

[com1DFA_com1DFA_defaultConfig]
# Use defaults for other parameters
"""
    return config_content


def setup_simulation_folder(output_folder: Path, 
                            dem_path: Path, 
                            release_path: Path,
                            mu: float, 
                            xi: float) -> Path:
    """
    Create a simulation folder with the AVAFRAME expected structure.
    
    Returns the path to the created simulation folder.
    """
    # Create folder name
    folder_name = f"mu_{mu:.3f}_turb_{xi:.0f}"
    sim_folder = output_folder / folder_name
    
    # Create directory structure
    inputs_folder = sim_folder / "Inputs"
    rel_folder = inputs_folder / "REL"
    
    for folder in [sim_folder, inputs_folder, rel_folder]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Copy DEM
    shutil.copy(dem_path, inputs_folder / "DEM.asc")
    
    # Copy projection file if exists
    prj_path = dem_path.with_suffix('.prj')
    if prj_path.exists():
        shutil.copy(prj_path, inputs_folder / "DEM.prj")
    
    # Copy release area
    rel_dest = rel_folder / "relTh.asc"
    shutil.copy(release_path, rel_dest)
    
    # Copy release projection if exists
    rel_prj = release_path.with_suffix('.prj')
    if rel_prj.exists():
        shutil.copy(rel_prj, rel_folder / "relTh.prj")
    
    # Create configuration file
    config_content = create_avaframe_config(mu, xi)
    config_path = sim_folder / "local_com6RockAvalancheCfg.ini"
    config_path.write_text(config_content)
    
    # Create placeholder folders that AVAFRAME expects
    for subfolder in ["ENT", "RES", "LINES", "POINTS", "POLYGONS"]:
        (inputs_folder / subfolder).mkdir(exist_ok=True)
    
    return sim_folder


def run_avaframe_simulation(sim_folder: Path, avaframe_path: Path) -> bool:
    """
    Run AVAFRAME simulation for the given folder.
    
    Returns True if successful, False otherwise.
    """
    # Create runner script
    runner_script = f'''
import sys
sys.path.insert(0, "{avaframe_path}")

from avaframe.com6RockAvalanche import com6RockAvalanche
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils

# Initialize
avalancheDir = "{sim_folder}"
logUtils.initiateLogger(avalancheDir)

# Load configuration
cfg = cfgUtils.getModuleConfig(com6RockAvalanche, fileOverride="{sim_folder / 'local_com6RockAvalancheCfg.ini'}")

# Run simulation
com6RockAvalanche.runCom6RockAvalanche(cfg, avalancheDir)
'''
    
    runner_path = sim_folder / "run_simulation.py"
    runner_path.write_text(runner_script)
    
    # Run the simulation
    try:
        result = subprocess.run(
            ["python", str(runner_path)],
            capture_output=True,
            text=True,
            cwd=str(sim_folder),
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"  Error: {result.stderr[:500]}")
            return False
        return True
        
    except subprocess.TimeoutExpired:
        print("  Timeout!")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run AVAFRAME rock avalanche parameter sweep"
    )
    
    # Input files
    parser.add_argument("--dem", required=True, type=Path,
                       help="Path to DEM file (ASC format)")
    parser.add_argument("--rel", required=True, type=Path,
                       help="Path to release thickness file (ASC format)")
    
    # Friction coefficient (μ) range
    parser.add_argument("--mu-min", type=float, default=0.025,
                       help="Minimum friction coefficient (default: 0.025)")
    parser.add_argument("--mu-max", type=float, default=0.4,
                       help="Maximum friction coefficient (default: 0.4)")
    parser.add_argument("--mu-steps", type=int, default=16,
                       help="Number of friction coefficient steps (default: 16)")
    
    # Turbulence coefficient (ξ) range
    parser.add_argument("--xi-min", type=float, default=250,
                       help="Minimum turbulence coefficient (default: 250)")
    parser.add_argument("--xi-max", type=float, default=2000,
                       help="Maximum turbulence coefficient (default: 2000)")
    parser.add_argument("--xi-steps", type=int, default=8,
                       help="Number of turbulence coefficient steps (default: 8)")
    
    # Output
    parser.add_argument("--output", "-o", type=Path, required=True,
                       help="Output folder for simulations")
    
    # AVAFRAME path
    parser.add_argument("--avaframe", type=Path, 
                       default=Path("/app/AvaFrame"),
                       help="Path to AVAFRAME installation")
    
    # Options
    parser.add_argument("--dry-run", action="store_true",
                       help="Only create folders, don't run simulations")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip simulations that already have results")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.dem.exists():
        print(f"Error: DEM file not found: {args.dem}")
        return 1
    
    if not args.rel.exists():
        print(f"Error: Release file not found: {args.rel}")
        return 1
    
    # Create output folder
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter grid
    mu_values = np.linspace(args.mu_min, args.mu_max, args.mu_steps)
    xi_values = np.linspace(args.xi_min, args.xi_max, args.xi_steps)
    
    total_sims = len(mu_values) * len(xi_values)
    print(f"\n{'='*60}")
    print("AVAFRAME Rock Avalanche Parameter Sweep")
    print(f"{'='*60}")
    print(f"DEM: {args.dem}")
    print(f"Release: {args.rel}")
    print(f"μ range: {args.mu_min:.3f} - {args.mu_max:.3f} ({len(mu_values)} steps)")
    print(f"ξ range: {args.xi_min:.0f} - {args.xi_max:.0f} ({len(xi_values)} steps)")
    print(f"Total simulations: {total_sims}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Save sweep configuration
    sweep_config = {
        "dem": str(args.dem),
        "release": str(args.rel),
        "mu_values": mu_values.tolist(),
        "xi_values": xi_values.tolist(),
        "total_simulations": total_sims,
        "created": datetime.now().isoformat(),
    }
    config_path = args.output / "sweep_config.json"
    config_path.write_text(json.dumps(sweep_config, indent=2))
    
    # Run simulations
    completed = 0
    failed = 0
    skipped = 0
    
    for i, mu in enumerate(mu_values):
        for j, xi in enumerate(xi_values):
            sim_num = i * len(xi_values) + j + 1
            print(f"[{sim_num}/{total_sims}] μ={mu:.3f}, ξ={xi:.0f}", end=" ")
            
            # Setup folder
            sim_folder = setup_simulation_folder(
                args.output, args.dem, args.rel, mu, xi
            )
            
            # Check if results exist
            peak_files = list(sim_folder.glob("Outputs/com1DFA/peakFiles/*_pft.asc"))
            if peak_files and args.skip_existing:
                print("- skipped (exists)")
                skipped += 1
                continue
            
            if args.dry_run:
                print("- folder created (dry run)")
                continue
            
            # Run simulation
            print("- running...", end=" ", flush=True)
            success = run_avaframe_simulation(sim_folder, args.avaframe)
            
            if success:
                print("✓ done")
                completed += 1
            else:
                print("✗ failed")
                failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"\nResults saved to: {args.output}")
    print(f"Open with: streamlit run runout_explorer.py")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
