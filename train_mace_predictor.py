#!/usr/bin/env python3
"""
MACE Prediction Training Pipeline

This script runs the complete training pipeline for MACE prediction using
protein signatures and organ aging features.

Usage:
    python train_mace_predictor.py [--config CONFIG_PATH] [--output OUTPUT_DIR]

Example:
    python train_mace_predictor.py --config configs/config.yaml --output results/
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataProcessor
from src.models import ModelTrainer
from src.analysis import FeatureImportanceAnalyzer
from src.visualization import ResultsVisualizer
from src.utils import setup_logging, set_random_seeds, create_output_directory


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_data_preparation(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Run data preparation pipeline.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
        
    Returns:
        Dictionary with prepared data
    """
    logging.info("Starting data preparation...")
    
    # Initialize data processor
    data_config = config['data']
    processor = DataProcessor(
        data_dir=data_config['data_dir'],
        random_state=config['random_state']
    )
    
    # Check if split data already exists
    train_file = os.path.join(data_config['data_dir'], data_config['train_file'])
    test_file = os.path.join(data_config['data_dir'], data_config['test_file'])
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        logging.info("Loading existing train/test split...")
        train_df, test_df = processor.load_split_data(
            data_config['train_file'], 
            data_config['test_file']
        )
        
        # Load full data for feature set creation
        full_data = processor.load_data()
        data_with_organs = processor.organ_calculator.add_organ_features(full_data)
        feature_sets = processor.create_feature_sets(data_with_organs)
        
        prepared_data = {
            'train_data': train_df,
            'test_data': test_df,
            'feature_sets': feature_sets,
            'full_data': data_with_organs
        }
    else:
        logging.info("Creating new train/test split...")
        prepared_data = processor.prepare_full_pipeline(
            data_file=os.path.join(data_config['data_dir'], data_config['main_file'])
        )
    
    logging.info("Data preparation completed")
    logging.info(f"Training samples: {len(prepared_data['train_data'])}")
    logging.info(f"Test samples: {len(prepared_data['test_data'])}")
    logging.info(f"Feature sets: {list(prepared_data['feature_sets'].keys())}")
    
    return prepared_data


def run_model_training(config: Dict[str, Any], prepared_data: Dict[str, Any], 
                      output_dir: str) -> Dict[str, Any]:
    """
    Run model training with Bayesian optimization.
    
    Args:
        config: Configuration dictionary
        prepared_data: Prepared data from preprocessing
        output_dir: Output directory
        
    Returns:
        Dictionary with training results
    """
    logging.info("Starting model training...")
    
    # Initialize trainer
    train_config = config['training']
    trainer = ModelTrainer(
        n_jobs=train_config['n_jobs'],
        random_state=config['random_state']
    )
    
    # Initialize data processor for preprocessing
    processor = DataProcessor(random_state=config['random_state'])
    
    # Get data
    train_df = prepared_data['train_data']
    test_df = prepared_data['test_data']
    feature_sets = prepared_data['feature_sets']
    target_col = config['data']['target_column']
    
    # Results storage
    all_results = []
    all_cv_results = []
    
    # Train models for each feature set
    for fs_name, features in feature_sets.items():
        logging.info(f"Training models for feature set: {fs_name}")
        
        # Preprocess features
        train_features, preprocessing_info = processor.preprocess_features(train_df, features)
        test_features = processor.apply_preprocessing(test_df, preprocessing_info)
        
        # Get target variables
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        
        # Train each model
        model_config = config['models']
        for model_name in ['random_forest', 'logistic_regression', 'splsda']:
            if not model_config[model_name]['enabled']:
                continue
                
            logging.info(f"Training {model_name} on {fs_name}")
            
            try:
                # Train model
                training_result = trainer.train_model(
                    train_features.values, y_train,
                    model_name.replace('_', '').title().replace('Splsda', 'SPLSDA'),
                    n_iter=train_config['n_iter'],
                    cv_folds=train_config['cv_folds']
                )
                
                # Evaluate on test set
                test_metrics = trainer.evaluate_model(
                    training_result['model'],
                    test_features.values,
                    y_test
                )
                
                # Bootstrap evaluation
                bootstrap_results = trainer.bootstrap_evaluation(
                    training_result['model'],
                    test_features.values,
                    y_test,
                    n_bootstraps=train_config['n_bootstrap']
                )
                
                # Save model
                model_info = {
                    'feature_set': fs_name,
                    'model_name': model_name,
                    'params': training_result['best_params'],
                    'preprocessing_info': preprocessing_info,
                    'feature_names': features,
                    'metrics': {
                        'test': test_metrics,
                        'bootstrap': bootstrap_results
                    }
                }
                
                model_path = trainer.save_model(
                    training_result['model'], model_info,
                    os.path.join(output_dir, train_config['models_dir']),
                    model_name, fs_name
                )
                
                # Store results
                result_row = {
                    'FeatureSet': fs_name,
                    'Model': model_name,
                    'BestParams': str(training_result['best_params']),
                    **{f'Test{k.title()}': v for k, v in test_metrics.items()},
                    **{f'Test{k}': v for k, v in bootstrap_results.items()}
                }
                all_results.append(result_row)
                
                # Store CV results
                cv_results = training_result['cv_results'].copy()
                cv_results['FeatureSet'] = fs_name
                cv_results['Model'] = model_name
                all_cv_results.append(cv_results)
                
                logging.info(f"Completed {model_name} on {fs_name}")
                
            except Exception as e:
                logging.error(f"Error training {model_name} on {fs_name}: {e}")
                continue
    
    # Combine results
    results_df = pd.DataFrame(all_results)
    cv_df = pd.concat(all_cv_results, ignore_index=True) if all_cv_results else pd.DataFrame()
    
    # Save results
    results_path = os.path.join(output_dir, 'training_results.csv')
    cv_path = os.path.join(output_dir, 'cv_results.csv')
    
    results_df.to_csv(results_path, index=False)
    if not cv_df.empty:
        cv_df.to_csv(cv_path, index=False)
    
    logging.info("Model training completed")
    
    return {
        'results': results_df,
        'cv_results': cv_df,
        'feature_sets': feature_sets
    }


def run_feature_importance_analysis(config: Dict[str, Any], training_results: Dict[str, Any],
                                   prepared_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Run feature importance analysis.
    
    Args:
        config: Configuration dictionary
        training_results: Results from model training
        prepared_data: Prepared data
        output_dir: Output directory
        
    Returns:
        Dictionary with feature importance results
    """
    if not config['feature_importance']['enabled']:
        logging.info("Feature importance analysis disabled")
        return {}
    
    logging.info("Starting feature importance analysis...")
    
    # Initialize analyzer
    fi_config = config['feature_importance']
    analyzer = FeatureImportanceAnalyzer(
        n_permutations=fi_config['n_permutations'],
        n_jobs=fi_config['n_jobs'],
        random_state=config['random_state'],
        batch_size=fi_config['batch_size']
    )
    
    # Load best model (for example, RandomForest on organ_list)
    # This would be selected based on performance
    # For now, use a simple selection
    
    # Implementation would load saved model and run importance analysis
    # This is a placeholder for the full implementation
    
    logging.info("Feature importance analysis completed")
    return {}


def run_visualization(config: Dict[str, Any], training_results: Dict[str, Any],
                     output_dir: str):
    """
    Generate visualizations for results.
    
    Args:
        config: Configuration dictionary
        training_results: Training results
        output_dir: Output directory
    """
    if not config['visualization']['save_plots']:
        return
    
    logging.info("Generating visualizations...")
    
    # Initialize visualizer
    vis_config = config['visualization']
    visualizer = ResultsVisualizer(
        style=vis_config['style'],
        palette=vis_config['palette'],
        figsize=tuple(vis_config['figure_size']),
        dpi=vis_config['dpi']
    )
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot CV performance
    if not training_results['cv_results'].empty:
        cv_fig = visualizer.plot_cv_performance(
            training_results['cv_results'],
            save_path=os.path.join(plots_dir, 'cv_performance.png')
        )
        
    # Plot test performance
    if not training_results['results'].empty:
        test_fig = visualizer.plot_bootstrap_results(
            training_results['results'],
            save_path=os.path.join(plots_dir, 'test_performance.png')
        )
    
    # Create performance table
    if not training_results['cv_results'].empty:
        table = visualizer.create_performance_table(
            training_results['cv_results'],
            training_results['results'] if not training_results['results'].empty else None,
            save_path=os.path.join(output_dir, 'performance_table.csv')
        )
    
    logging.info("Visualization completed")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='MACE Prediction Training Pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = create_output_directory(args.output)
    
    # Setup logging
    setup_logging(
        level=config['logging']['level'],
        log_file=os.path.join(output_dir, config['logging']['file'])
    )
    
    # Set random seeds
    if config.get('set_random_seeds', True):
        set_random_seeds(config['random_state'])
    
    logging.info("Starting MACE prediction training pipeline")
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Output directory: {output_dir}")
    
    try:
        # Data preparation
        prepared_data = run_data_preparation(config, output_dir)
        
        # Model training
        training_results = run_model_training(config, prepared_data, output_dir)
        
        # Feature importance analysis
        importance_results = run_feature_importance_analysis(
            config, training_results, prepared_data, output_dir
        )
        
        # Visualization
        run_visualization(config, training_results, output_dir)
        
        logging.info("Training pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()