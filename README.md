# MACE Prediction Using Protein Signatures and Organ Aging

This repository contains the code and analysis for predicting Major Adverse Cardiovascular Events (MACE) using protein expression profiles and organ aging signatures.

## Overview

This study develops machine learning models to predict MACE using:
- Protein expression data from SomaScan platform
- Organ aging signatures computed using OrganAge library
- Multiple machine learning approaches with Bayesian optimization
- Comprehensive feature importance and biological pathway analysis

## Repository Structure

```
├── src/                    # Source code
│   ├── models/            # Machine learning models
│   ├── preprocessing/     # Data preprocessing utilities
│   ├── analysis/          # Analysis scripts
│   ├── visualization/     # Plotting and visualization
│   └── utils/            # Utility functions
├── data/                  # Data files (not included for privacy)
├── results/              # Analysis results and figures
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit tests
├── configs/              # Configuration files
└── requirements.txt      # Python dependencies
```

## Key Features

- **Multi-modal Prediction**: Combines protein expression and organ aging features
- **Bayesian Optimization**: Automated hyperparameter tuning for optimal performance
- **Feature Importance Analysis**: Comprehensive analysis of protein contributions
- **Biological Pathway Enrichment**: Disease pathway analysis of important proteins
- **Reproducible Pipeline**: Structured code with clear separation of concerns

## Requirements

- Python 3.8+
- OrganAge library
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- scikit-optimize
- gseapy

## Usage

1. **Data Preprocessing**:
```python
from src.preprocessing import DataProcessor
processor = DataProcessor()
train_data, test_data = processor.split_and_prepare_data()
```

2. **Model Training**:
```python
from src.models import MACEPredictor
predictor = MACEPredictor()
predictor.train_with_bayesian_optimization(train_data)
```

3. **Feature Importance Analysis**:
```python
from src.analysis import FeatureImportanceAnalyzer
analyzer = FeatureImportanceAnalyzer()
importance_results = analyzer.calculate_importance(model, test_data)
```

4. **Visualization and Results**:
```python
from src.visualization import ResultsVisualizer
visualizer = ResultsVisualizer()
visualizer.plot_performance_metrics(results)
```

## Citation

If you use this code in your research, please cite our paper:

```
[Paper citation will be added upon publication]
```

## License

This code is provided for research purposes. Data files are not included due to patient privacy considerations.

## Contact

For questions about the code or methodology, please contact [contact information].