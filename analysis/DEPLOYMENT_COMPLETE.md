# ✅ Rock Avalanche Runout Explorer - Deployment Complete

## Status: READY TO USE

The full Rock Avalanche Runout Explorer has been successfully deployed with all features from your original implementation.

### What's Deployed

**Files:**
- `runout_explorer.py` (1,110 lines) - Full visualization tool
- `run_parameter_sweep.py` (301 lines) - Batch simulation runner  
- `requirements.txt` - Dependencies (all installed)
- `README.md` - Complete documentation
- `launch_explorer.sh` - Quick launch script

### Features Included

✅ **Interactive Map Visualization**
- Hillshade terrain rendering
- Flow thickness overlay (rainbow colormap)
- Hmax/Hmin elevation markers
- Click coordinates display

✅ **Empirical Validation**
- **Strom et al. (2019)**: Three confinement types
  - Frontally confined (R² = 0.9258)
  - Laterally confined (R² = 0.9267)
  - Unconfined (R² = 0.9361)
  
- **Brideau et al. (2021)**: Two populations
  - Small/medium events (R² = 0.113, N=144)
  - Large events (R² = 0.431, N=288)

✅ **Parameter Space Analysis**
- Interactive heatmap (μ vs ξ)
- Color-coded by Strom absolute error
- Click to select simulations
- Slider navigation

✅ **Metrics Dashboard**
- Volume, Height drop, Runout length
- H/L mobility ratio
- Total affected area
- Strom AE (absolute error)

✅ **Authentication Integration**
- Works standalone OR with main app login
- Session state aware
- Graceful fallback if auth unavailable

### How to Use

**Option 1: Quick Launch**
```bash
cd /home/gustav/avalanche-app/analysis
./launch_explorer.sh
```

**Option 2: Manual Launch**
```bash
cd /home/gustav/avalanche-app
source venv/bin/activate
streamlit run analysis/runout_explorer.py
```

**Option 3: From Main App**
After logging into `app/main_map_new.py`, open in new terminal:
```bash
streamlit run analysis/runout_explorer.py --server.port 8502
```
(Authentication will be recognized across ports)

### Expected Data Structure

Point the tool to a directory containing parameter sweep results:

```
/mnt/data/simulations/
├── mu_0.025_turb_250/
│   ├── Inputs/
│   │   ├── DEM.asc
│   │   └── REL/relTh.asc
│   └── Outputs/com1DFA/peakFiles/
│       └── *_pft.asc
├── mu_0.025_turb_500/
├── mu_0.050_turb_250/
└── ...
```

Or a single simulation folder with the same structure.

### Running Parameter Sweeps

To generate new sweeps for the explorer:

```bash
python analysis/run_parameter_sweep.py \
    --dem /path/to/DEM.asc \
    --rel /path/to/release.asc \
    --mu-min 0.025 --mu-max 0.4 --mu-steps 16 \
    --xi-min 250 --xi-max 2000 --xi-steps 8 \
    --output /mnt/data/my_parameter_sweep
```

This creates 128 simulations (16 × 8 grid) ready for exploration.

### Key Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **AE_Strom** | Absolute error from Strom's relation | < 0.2 = good match |
| **H/L** | Mobility ratio | Lower = more mobile |
| **V×H** | Volume × Height product | Input to Strom relation |
| **A_total** | Planimetric area affected | Compare to empirical |

### Troubleshooting

**"No simulations found"**
- Check folder names match pattern: `mu_X.XXX_turb_XXXX`
- Ensure `Outputs/com1DFA/peakFiles/*_pft.asc` exists

**Memory issues**
- Reduce DEM resolution
- Process fewer simulations at once
- Increase available RAM

**Authentication not recognized**
- Ensure main app is running first
- Check same browser/session
- Try standalone mode (works without auth)

### References

1. **Strom, A., L. Li, and H. Lan. 2019.** "Rock Avalanche Mobility: Optimal Characterization and the Effects of Confinement." *Landslides* 16(8): 1437-52.

2. **Brideau, M.-A., et al. 2021.** Version 2.2, 28 Jan 2020. Global landslide database with H/L vs Volume relationship.

3. **AVAFRAME Documentation:** https://docs.avaframe.org/

---

**Deployed:** 26 January 2026  
**Version:** Full implementation (1110 lines)  
**Status:** Production ready ✅
