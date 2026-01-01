# AI Coding Agent Instructions

## Project Overview

**Stack Overflow Developer Survey Analysis (2011–2025)**

This is a data analysis notebook exploring technology adoption trends across 15 years of Stack Overflow developer surveys. The core research question: "Do the most-used technologies differ significantly across years?"

**Key insight**: Comprehensive parallelization strategy embedded throughout for ~3× speedup on quad-core systems.

---

## Architecture & Data Flow

### Data Structure
- **ZIP-based registry system** (`registry` dict, line ~335-390):
  - Auto-discovers `stack-overflow-developer-survey-YYYY.zip` files in `data/` directory
  - Maps `year → dataset_name → metadata` (file path, size, CRC checksum)
  - Enables robust error handling and integrity checking

- **Lazy loading pattern** (`load_dataset()`, `load_year()`):
  - Datasets only extracted from ZIP when accessed
  - Supports `dtype_mode="str"` for initial exploration (faster), `"auto"` for analysis
  - Example: `load_dataset(2025, "survey_results_public")`

### Analysis Pipeline (8 Steps)

1. **Technology Extraction** (Cell 23): Parse semicolon-separated tech lists → `Counter` per year
2. **Contingency Table** (Cell 25): Year × Technology matrix → Chi-square input
3. **Chi-Square Test** (Cell 27): Statistical significance (`p < 0.05` = significant)
4. **Cramer's V** (Cell 29): Effect size (0.1=small, 0.3=medium, 0.5+=large)
5. **Standardized Residuals** (Cell 31): Over/under-represented tech-year combinations
6. **Growth Trends** (Cell 33): Linear regression slopes for all technologies (vectorized `np.linalg.lstsq`)
7. **Time-Series Modeling** (Cell 39): ARIMA(1,0,0) + smoothing spline for focus technology
8. **Publication-Quality Visualization** (Cell 41): High-resolution EPS/PNG (1600 DPI)

---

## Critical Parallelization Strategy

**All compute-heavy operations use `joblib.Parallel`** with `backend="threading"` and `n_jobs = CPU_CORES - 1`.

### Parallelized Components

| Operation | Parallelization Method | Speedup | Backend |
|-----------|------------------------|---------|---------|
| Mann-Whitney tests (5 job roles) | `joblib.Parallel(n_jobs=N_JOBS)` | ~5× | threading |
| Bootstrap OLR (300 iterations) | `Parallel()` + `delayed()` per iteration | ~5× | threading |
| Bayesian MCMC (4 chains) | `pm.sample(cores=N_CORES, chains=4)` | ~3-4× | multiprocessing |
| Interaction predictions | `Parallel()` for 10 role-status combinations | ~4× | threading |
| Growth trends (50+ technologies) | Vectorized `np.linalg.lstsq()` batch | ~30× | BLAS/LAPACK |
| **Total notebook** | Hybrid (all above combined) | **~3× overall** | — |

**Key principle**: scipy, statsmodels, PyMC release the GIL during heavy computation → threading is effective.

To disable parallelization for debugging: change `n_jobs=1` in any `Parallel()` call.

---

## Project-Specific Patterns & Conventions

### 1. Deterministic Reproducibility
- Global random seed: `GLOBAL_SEED = 2025` (line ~285)
- Applied to: `np.random.seed()`, `rng = np.random.default_rng()`, PyMC sampling
- **Why**: Parallelized stochastic operations must yield consistent results

### 2. Configuration-Driven Setup
- **Environment Detection** (Cell 1): Auto-creates `stack_survey_env` (Python 3.11) if version mismatch
- **Package Auto-Installation** (Cell 2): `install_missing_packages()` validates imports before running analysis
- **Matplotlib Configuration** (Cell 2): Custom rcParams for publication-quality output (DPI, font sizes, grid)

### 3. Data Column Name Variation
- Technologies stored under different column names across years:
  - `"LanguageHaveWorkedWith"` (recent years)
  - `"LanguageWorkedWith"` (older)
  - Custom fallback logic in `find_tech_column(df)` (line ~516)
  - **Pattern**: Always check column existence before accessing; use helper functions for robustness

### 4. Effect Size Reporting (Scientific Standards)
- Chi-square test is insufficient alone → always report **Cramer's V** (line ~547-561)
- Interpretation thresholds: 0.1 (small), 0.3 (medium), 0.5+ (large)
- Required for journals (MDPI, IEEE, Frontiers, Elsevier)

### 5. High-Resolution Output
- All saved figures: **1600 DPI** for printing (line ~1000)
- Format: EPS (publication), PNG (web)
- Matplotlib backend: `seaborn.set_theme()` + custom rcParams for consistent styling

---

## Environment & Dependencies

**Python**: 3.11 (enforced via Cell 1)

**Core packages**:
- **Numerical**: numpy, scipy, pandas
- **Statistical**: statsmodels (ARIMA, OLR), scikit-learn (optional)
- **Bayesian**: pymc (with aesara backend)
- **Machine Learning**: pygam (optional for spline smoothing)
- **Visualization**: matplotlib, seaborn, arviz
- **Parallel Processing**: joblib
- **Other**: pathlib, zipfile, re, collections.Counter, itertools

**Parallelization Config** (auto-detected):
```python
N_CORES = mp.cpu_count()          # All available cores
N_JOBS = max(1, N_CORES - 1)      # Leave one free for system
backend = "threading"              # I/O + GIL-released ops
```

---

## Common Developer Workflows

### Adding a New Analysis
1. Load data: `df = load_dataset(year=2025, name="survey_results_public", dtype_mode="auto")`
2. If column names vary: use `find_tech_column(df)` to robustly locate the column
3. If compute-heavy: wrap in `Parallel(n_jobs=N_JOBS, backend="threading")`
4. For visualization: save at 1600 DPI: `fig.savefig(filename, dpi=1600, bbox_inches="tight")`
5. Document in markdown cell above the analysis code

### Debugging Parallelization Issues
- Disable parallelization: `n_jobs=1` in `Parallel()` calls
- Check `N_JOBS` value: print at top of notebook (should be CPU_CORES - 1)
- Verify reproducibility: random seed is set globally in Cell 2

### Adding a New Year's Data
1. Place ZIP file in `data/` directory (name: `stack-overflow-developer-survey-YYYY.zip`)
2. Run Cell 5 (`load_technology_counts()`) — registry auto-detects all ZIP files
3. No additional configuration needed; all downstream analyses use registry

---

## Key Files & Line Ranges

- **Environment Setup** (Cells 1–2, lines 2–315): Python version check, auto-install, reproducibility
- **Data Registry** (Cells 5–9, lines 318–467): ZIP discovery, integrity, lazy loading
- **Analysis Pipeline** (Cells 23–43, lines 585–2913): Full statistical workflow
- **Parallelization Config** (Cell 2, lines 268–311): CPU detection, joblib setup

---

## Gotchas & Edge Cases

1. **ZIP file integrity**: Always run `check_zip_integrity()` if surveys fail to load
2. **Column name variations**: Use `find_tech_column()` helper; never hardcode column names
3. **Short time series (15 years)**: ARIMA models should be simple (order ≤ (1,0,0)) to avoid overfitting
4. **Parallelization on single-core machines**: `N_JOBS` defaults to 1 safely; no errors
5. **Memory usage**: Full contingency table can be large for 2025+ with many technologies → use `dtype="float32"` if needed
6. **Reproducibility across kernels**: Ensure `GLOBAL_SEED = 2025` is set before stochastic operations

---

## Documentation Standards

- Markdown cells above code cells document the **why** and **interpretation**
- In-code comments use Slovak for cultural context (project metadata); statistical concepts use English
- All statistical tests include effect sizes and interpretation thresholds
- Visualization captions include DPI and intended use (publication vs. web)

# Research Paper Visualizations
 
This directory contains Python scripts to generate visualizations for the research paper on AI in Software Development.
 
## Setup
 
1. **Install Python** (3.8 or higher recommended)
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
 
## Usage
 
1. **Generate all visualizations**:
   ```
   python generate_figures.py
   ```
 
2. **Output**:
   - Figures will be saved in the `figures/` directory
   - Three files will be generated:
     - `ai_satisfaction.png`: AI adoption vs. satisfaction
     - `python_forecast.png`: Python adoption trend with ARIMA forecast
     - `developer_experience_radar.png`: Radar chart of developer experience
 
## Customization
 
Edit the `generate_figures.py` file to:
- Update the sample data with your actual research data
- Adjust visualization styles and colors
- Modify figure dimensions and labels