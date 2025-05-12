"""
Weights & Biases hyperparameter sweep configuration for Rock-Paper-Scissors hand gesture classifier.
This script configures and launches a hyperparameter sweep using Weights & Biases.
Run this script with 'python rps_sweep.py' to start a sweep.
"""

import wandb
import subprocess
import argparse
import os
import sys
import time

# Default constants
DEFAULT_SWEEP_COUNT = 20
DEFAULT_TEAM_NAME = "praise_mlops"
DEFAULT_PROJECT_NAME = "rock-paper-scissors"
DEFAULT_PROGRAM_NAME = "train.py"

def create_sweep_config(program_name):
    """Create the sweep configuration dictionary."""
    print("📋 Creating sweep configuration...")
    sweep_config = {
        'method': 'bayes',  # Grid, random, or bayesian optimization
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'sweep': {
                'value': True
            },
            # Model architecture parameters
            'hidden_sizes': {
                'values': [
                    '16',
                    '32',
                    '64',
                    '128',
                    '32,16',
                    '64,32',
                    '128,64',
                    '64,32,16',
                    '128,64,32'
                ]
            },
            'activation': {
                'values': ['relu', 'leaky_relu', 'elu', 'gelu']
            },
            'dropout_rate': {
                'values': [0.0, 0.1, 0.2, 0.3, 0.5]
            },
            'batch_norm': {
                'values': [True, False]
            },
            'residual': {
                'values': [True, False]
            },
            
            # Training parameters
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
            },
            'optimizer': {
                'values': ['adam', 'sgd', 'rmsprop']
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-3
            },
            
            # Learning rate scheduler
            'scheduler': {
                'values': [None, 'step', 'cosine', 'plateau']
            },
            
            # Feature selection
            'feature_groups': {
                'values': [
                    'all',  # Use all features
                    'extension,angles',  # Extension ratios and angles only
                    'extension,distances',  # Extension ratios and inter-finger distances
                    'extension,opposition',  # Extension ratios and thumb opposition
                    'angles,distances',  # Angles and distances
                    'extension,angles,distances',  # All except opposition
                    'extension,angles,opposition'  # All except distances
                ]
            }
        },
        'program': program_name,
    }
    return sweep_config

def run_sweep_agent(sweep_id, count, team=None, project=None):
    """Run the sweep agent with proper error handling."""
    # Construct the sweep command
    if team:
        sweep_command = f"wandb agent {team}/{project}/{sweep_id}"
    else:
        sweep_command = f"wandb agent {sweep_id}"
    
    # Add count parameter
    sweep_command += f" --count {count}"
    
    print(f"🚀 Running sweep agent with command: {sweep_command}")
    print(f"🔄 Starting {count} sweep runs...")
    
    start_time = time.time()
    try:
        # Run the sweep agent
        result = subprocess.run(
            sweep_command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        elapsed = time.time() - start_time
        print(f"✅ Sweep agent completed successfully in {elapsed:.1f}s")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Error running sweep agent after {elapsed:.1f}s: {e}")
        print(f"📤 Command output: {e.stdout}")
        print(f"⚠️ Command error: {e.stderr}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Unexpected error running sweep agent after {elapsed:.1f}s: {e}")
        return False

def main():
    """Main function to configure and run the sweep."""
    print("🚀 Starting W&B Hyperparameter Sweep Configuration")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Configure and run W&B hyperparameter sweep')
    parser.add_argument('--team', type=str, default=DEFAULT_TEAM_NAME, help='W&B team name')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT_NAME, help='W&B project name')
    parser.add_argument('--count', type=int, default=DEFAULT_SWEEP_COUNT, help='Number of sweep runs')
    parser.add_argument('--program', type=str, default=DEFAULT_PROGRAM_NAME, 
                        help='Training program to run (default: train.py)')
    parser.add_argument('--feature_groups', action='store_true', 
                        help='Enable all feature group combinations in the sweep')
    args = parser.parse_args()
    
    # Verify that the training program exists
    if not os.path.exists(args.program):
        print(f"❌ Error: Training program '{args.program}' not found.")
        print(f"⚠️ Make sure '{args.program}' exists in the current directory.")
        sys.exit(1)
    
    print(f"✅ Found training program: {args.program}")
    
    # Create sweep configuration
    sweep_config = create_sweep_config(args.program)
    
    # If feature_groups flag is not set, limit feature group options to just 'all'
    if not args.feature_groups:
        print("ℹ️ Using only 'all' for feature groups (use --feature_groups to enable all combinations)")
        sweep_config['parameters']['feature_groups']['values'] = ['all']
    else:
        print("🔍 Using all feature group combinations in sweep")
    
    print(f"\n📋 Sweep configuration summary:")
    print(f"   🔹 Method: {sweep_config['method']}")
    print(f"   🔹 Metric: {sweep_config['metric']['name']} (goal: {sweep_config['metric']['goal']})")
    print(f"   🔹 Program: {sweep_config['program']}")
    print(f"   🔹 Number of runs: {args.count}")
    print(f"   🔹 Team: {args.team if args.team else 'None (using default user)'}")
    print(f"   🔹 Project: {args.project}")
    print(f"   🔹 Feature combinations: {len(sweep_config['parameters']['feature_groups']['values'])}")
    print(f"   🔹 Total parameter combinations: {len(sweep_config['parameters']['hidden_sizes']['values']) * len(sweep_config['parameters']['activation']['values']) * len(sweep_config['parameters']['dropout_rate']['values']) * len(sweep_config['parameters']['batch_norm']['values']) * len(sweep_config['parameters']['residual']['values']) * len(sweep_config['parameters']['batch_size']['values']) * len(sweep_config['parameters']['optimizer']['values']) * len(sweep_config['parameters']['scheduler']['values']) * len(sweep_config['parameters']['feature_groups']['values'])}")
    
    try:
        print("\n🔄 Creating sweep on Weights & Biases...")
        start_time = time.time()
        
        # Initialize wandb with team and project
        if args.team:
            sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.team)
        else:
            sweep_id = wandb.sweep(sweep_config, project=args.project)
        
        print(f"✅ Sweep created with ID: {sweep_id} in {time.time() - start_time:.2f}s")
        
        # Run the sweep agent
        success = run_sweep_agent(sweep_id, args.count, args.team, args.project)
        
        if success:
            print(f"\n🎉 Sweep completed successfully!")
            print(f"📊 View results at: https://wandb.ai/{args.team or 'your-username'}/{args.project}/sweeps/{sweep_id}")
        else:
            print(f"\n⚠️ Sweep encountered errors. Check the logs above for details.")
            print(f"💡 You can manually restart the sweep with: wandb agent {args.team+'/' if args.team else ''}{args.project}/{sweep_id}")
    
    except Exception as e:
        print(f"❌ Error creating or running sweep: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()