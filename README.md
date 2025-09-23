# Preferences Networks 🔍

A sophisticated network analysis framework for studying preference formation in academic environments, implemented as part of my master's thesis research. The framework provides deep insights into social network formation and preference patterns across academic programs.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-blue.svg)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.0+-blue.svg)](https://pandas.pydata.org/)
[![CVXPY](https://img.shields.io/badge/CVXPY-1.0+-blue.svg)](https://www.cvxpy.org/)

## 🎯 Key Features

- **Dual Model Implementation**:
  - 📊 Simple Case ($L=D=1$, $|X|=2$): Binary characteristic analysis
  - 🔄 Generalized Case ($|X|=4$): Multi-characteristic modeling

- **Advanced Analytics**:
  - Network structure analysis
  - Preference pattern identification
  - Counterfactual scenario generation
  - Parameter space exploration

## 📚 Documentation

- [📖 Detailed Thesis Documentation](./Anexos_tesis.pdf)
- [⚙️ Configuration Guide](#configuration)
- [📊 Analysis Pipeline](#analysis-pipeline)
- [� Results & Visualization](#results)

## 🚀 Quick Start

1. **Clone & Setup**:
   ```bash
   git clone https://github.com/santineiraher/preferences_networks.git
   cd preferences_networks
   pip install -r requirements.txt
   ```

2. **Configure Paths** in `config.py`:
   ```python
   DATAFRAME_CSV_PATH = "path/to/your/data.csv"
   ```

3. **Run Analysis**:
   ```bash
   python src/Simple_mod/main.py  # For simple model
   # or
   python src/Generalized/orquestrator_qp_problem.py  # For generalized model
   ```

## 🧮 Mathematical Framework
<a name="math"></a>

### Simple Model ($L=D=1$, $|X|=2$)
- **Equilibrium Equations**:
  ```python
  s_BW_lo = min(p_BW * (1 - p_BB), ratio_WB * p_WB * (1 - p_WW))
  s_WB_lo = min(p_WB * (1 - p_WW), ratio_BW * p_BW * (1 - p_BB))
  ```

### Generalized Model ($|X|=4$)
- **Quadratic Programming**:
  - Uses CVXPY for optimization
  - Implements preference class constraints
  - Handles multi-dimensional characteristic space

## 🔄 Analysis Pipeline
<a name="analysis-pipeline"></a>

### 1. Simple Model Components

#### a. Network Construction
- **Input Processing**: [`dataframe_construction.py`]
  ```python
  def data_construction(df, term, program_title="", total=False):
      # Processes raw data into network-ready format
  ```

#### b. Parameter Analysis
- **Share Type Calculation**: [`helper_functions_simple.py`]
  ```python
  def share_typez(df, min_num, major, term):
      # Calculates type shares and distributions
  ```

#### c. Counterfactual Analysis
- **Link Pattern Analysis**: [`counterfactuals.py`]
  ```python
  def calculate_links(data, N_B, N_W):
      # Analyzes network connection patterns
  ```

### 2. Generalized Model Features

#### a. Advanced Optimization
- **QP Solver** [`qp_problem_new.py`]:
  - Matrix construction
  - Constraint handling
  - Preference class mapping

#### b. Room Assignment
- **Assignment Algorithm** [`Room_assignment.py`]:
  - Optimized group formation
  - Exposure balance
  - Constraint satisfaction

## 📊 Data Structure
<a name="data"></a>

```
src/
├── Simple_mod/           # Basic model implementation
│   ├── main.py          # Pipeline orchestration
│   ├── network_construction.py
│   └── counterfactuals.py
│
├── Generalized/         # Advanced model components
│   ├── qp_problem_new.py
│   ├── Room_assignment.py
│   └── Exposure_cons.py
│
└── utils/              # Shared utilities
    ├── helper_functions_simple.py
    └── generalized_utils.py
```

## 📈 Results & Visualization
<a name="results"></a>

The framework generates comprehensive analysis outputs:

- **Network Statistics**:
  - Cross-linkedness metrics
  - Type share distributions
  - Parameter estimations

- **Visualization**:
  ```python
  def plot_total_links_kde_with_stats(results_df, output_file=None):
      # Generates KDE plots with statistical annotations
  ```

Results are organized in:
```
data/
├── Results/
│   ├── analysis_results.csv   # Primary analysis outputs
│   ├── Factuals/             # Actual network patterns
│   └── Parameter_sets/       # Identified parameter spaces
└── Datasets/
    ├── Counterfactuals/      # Alternative scenarios
    ├── Networks_semesters_majors/
    └── Type_shares/          # Distribution analyses
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

### Areas for Enhancement
- [ ] Extended characteristic space
- [ ] Additional optimization methods
- [ ] Improved visualization tools

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citations

If you use this framework in your research, please cite:
```bibtex
@mastersthesis{preferences_networks,
  author = {Santiago Neira},
  title  = {Network Formation and Preference Analysis in Academic Environments},
  year   = {2023},
  school = {Your University}
}
```


