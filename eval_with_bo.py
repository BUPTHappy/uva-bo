"""
Evaluation script with Bayesian Optimization for Unified Video Action.
This script runs Bayesian optimization to find optimal parameters before evaluation.
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import json
import tempfile
import subprocess
import torch
import dill
import numpy as np
from typing import Dict, Any, Optional, Tuple
from omegaconf import OmegaConf, open_dict
from unified_video_action.optimization.bayesian_optimizer import UVABayesianOptimizer
import click


class EvaluationBayesianOptimizer:
    """
    Bayesian optimizer for evaluation phase.
    Runs optimization before final evaluation to find best parameters.
    """
    
    def __init__(self, 
                 max_trials: int = 20,
                 n_test: int = 3,
                 device: str = "cuda:0",
                 output_dir: str = "./bayesian_optimization_logs",
                 optimization_mode: str = "balanced"):
        """
        Initialize the evaluation Bayesian optimizer.
        
        Args:
            max_trials: Maximum number of optimization trials
            n_test: Number of test runs per evaluation
            device: Device for evaluation
            output_dir: Output directory for optimization logs
            optimization_mode: Optimization mode - "speed_priority", "performance_priority", or "balanced"
        """
        self.max_trials = max_trials
        self.n_test = n_test
        self.device = device
        self.output_dir = output_dir
        self.optimization_mode = optimization_mode
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = UVABayesianOptimizer(max_trials=max_trials, optimization_mode=optimization_mode)
        
        print(f"EvaluationBayesianOptimizer initialized:")
        print(f"  Max trials: {max_trials}")
        print(f"  N test: {n_test}")
        print(f"  Device: {device}")
        print(f"  Output dir: {output_dir}")
        print(f"  Optimization mode: {optimization_mode}")
    
    def evaluate_model_with_params(self, 
                                 params: Dict[str, Any], 
                                 checkpoint_path: str) -> float:
        """
        Evaluate model with given parameters.
        
        Args:
            params: Parameters to use for evaluation
            checkpoint_path: Path to checkpoint
            
        Returns:
            Evaluation score
        """
        try:
            print(f"\n{'='*60}")
            print(f"Evaluating params: {params}")
            print(f"{'='*60}")
            
            # Clear CUDA cache before evaluation
            torch.cuda.empty_cache()
            
            # Wait a bit to ensure checkpoint is fully written
            import time
            time.sleep(2)
            
            # Check if checkpoint file is valid before loading
            if not self._is_checkpoint_valid(checkpoint_path):
                print(f"Checkpoint file is invalid or corrupted: {checkpoint_path}")
                return -1000.0
            
            # Load checkpoint to get config info
            payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, weights_only=False)
            cfg = payload["cfg"]
            
            # Determine timeout based on task
            if "libero" in cfg.task.name:
                timeout = max(1200, self.n_test * 60 * 10)  # At least 20 minutes for libero
            else:
                timeout = max(300, self.n_test * 60)  # At least 5 minutes for other tasks
            
            # Create temp output directory
            temp_output_dir = tempfile.mkdtemp(prefix="bayesian_eval_")
            
            # Run evaluation with parameters
            cmd = [
                "python", "eval_sim.py",
                "--checkpoint", checkpoint_path,
                "--output_dir", temp_output_dir,
                "--device", self.device,
                "--num_sampling_steps", str(params['num_sampling_steps']),
                "--cfg", str(params['cfg']),
                "--temperature", str(params['temperature']),
                "--n_test", str(self.n_test)
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[-1] if ":" in self.device else "0"
            
            print(f"Running evaluation with timeout: {timeout} seconds")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
            )
            
            if result.returncode != 0:
                print(f"Evaluation failed: {result.stderr}")
                return -1000.0
            
            # Parse results
            eval_log_path = os.path.join(temp_output_dir, f'eval_log_{os.path.basename(checkpoint_path)}.json')
            
            if not os.path.exists(eval_log_path):
                print(f"Eval log not found: {eval_log_path}")
                return -1000.0
            
            with open(eval_log_path, 'r') as f:
                eval_results = json.load(f)
            
            # Extract score
            if "test_mean_score" in eval_results:
                score = eval_results["test_mean_score"]
            elif "test/pusht_mean_score" in eval_results:
                score = eval_results["test/pusht_mean_score"]
            elif "test/libero10_mean_score" in eval_results:
                score = eval_results["test/libero10_mean_score"]
            else:
                score_keys = [k for k in eval_results.keys() if "score" in k.lower()]
                if score_keys:
                    score = eval_results[score_keys[0]]
                else:
                    print(f"No score found, available keys: {list(eval_results.keys())}")
                    return -1000.0
            
            print(f"Score: {score}")
            
            # Cleanup
            subprocess.run(["rm", "-rf", temp_output_dir], check=False)
            
            return float(score)
            
        except subprocess.TimeoutExpired:
            print(f"Evaluation timeout after {timeout} seconds")
            print(f"Evaluation may need more time. Consider increasing timeout or reducing n_test.")
            torch.cuda.empty_cache()
            return -1000.0
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                print(f"CUDA error during evaluation: {e}")
                print("Clearing CUDA cache and retrying...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return -1000.0
            else:
                print(f"Runtime error during evaluation: {e}")
                return -1000.0
        except Exception as e:
            print(f"Evaluation error: {e}")
            torch.cuda.empty_cache()
            return -1000.0
    
    def run_optimization(self, checkpoint_path: str) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Run Bayesian optimization.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Tuple of (best parameters, best score) if optimization succeeded, None otherwise
        """
        print(f"\n{'='*60}")
        print(f"Starting Bayesian optimization")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        def objective_function(params):
            return self.evaluate_model_with_params(params, checkpoint_path)
        
        try:
            # Run optimization
            result = self.optimizer.optimize(objective_function)
            
            # Handle case where optimization returns None or fails
            if result is None:
                print("Optimization returned None - all trials failed")
                return None
            
            # Unpack result safely
            if isinstance(result, tuple) and len(result) == 2:
                best_params, best_score = result
            else:
                print(f"Unexpected optimization result format: {result}")
                return None
            
            if best_params is not None and best_score > -1000.0:
                print(f"\n{'='*60}")
                print(f"Optimization completed!")
                print(f"Best score: {best_score:.4f}")
                print(f"Best params: {best_params}")
                print(f"{'='*60}")
                
                # Save optimization results
                import time
                optimization_result = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'checkpoint_path': checkpoint_path,
                    'timestamp': time.time()
                }
                
                # Save to file
                results_file = os.path.join(self.output_dir, "optimization_results.json")
                with open(results_file, 'w') as f:
                    json.dump(optimization_result, f, indent=2)
                
                # Also save using optimizer's method
                self.optimizer.save_results(results_file)
                
                return (best_params, best_score)
            else:
                print("Optimization failed or no valid parameters found")
                return None
                
        except Exception as e:
            print(f"Optimization error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _is_checkpoint_valid(self, checkpoint_path: str) -> bool:
        """
        Check if checkpoint file is valid and not corrupted.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                return False
            
            # Check file size (should be reasonable)
            file_size = os.path.getsize(checkpoint_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                print(f"Checkpoint file too small: {file_size} bytes")
                return False
            
            # Try to actually load the checkpoint to verify it's valid
            try:
                with open(checkpoint_path, "rb") as f:
                    # Try to load just the header to check if it's valid
                    checkpoint = torch.load(f, pickle_module=dill, weights_only=False)
                    
                    # Check if it has the expected structure
                    if isinstance(checkpoint, dict) and 'cfg' in checkpoint:
                        return True
                    else:
                        print(f"Checkpoint doesn't have expected structure: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
                        return False
                        
            except Exception as load_error:
                print(f"Failed to load checkpoint: {load_error}")
                return False
                    
        except Exception as e:
            print(f"Error checking checkpoint validity: {e}")
            return False


@click.command()
@click.option("-c", "--checkpoint", required=True, help="Path to checkpoint file")
@click.option("-o", "--output_dir", required=True, help="Output directory for evaluation results")
@click.option("-d", "--device", default="cuda:0", help="Device for evaluation")
@click.option("--max_trials", type=int, default=20, help="Maximum number of optimization trials")
@click.option("--n_test", type=int, default=3, help="Number of test runs per evaluation")
@click.option("--optimization_mode", type=str, default="balanced", 
              help="Optimization mode: speed_priority, performance_priority, or balanced")
@click.option("--skip_optimization", is_flag=True, help="Skip optimization and run evaluation with default params")
def main(checkpoint, output_dir, device, max_trials, n_test, optimization_mode, skip_optimization):
    """
    Run evaluation with Bayesian optimization to find optimal parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if skip_optimization:
        print("Skipping optimization, running evaluation with default parameters...")
        # Run evaluation with default parameters
        cmd = [
            "python", "eval_sim.py",
            "--checkpoint", checkpoint,
            "--output_dir", output_dir,
            "--device", device,
            "--n_test", str(n_test)
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
        
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print("Evaluation failed")
            return
    else:
        # Run Bayesian optimization
        bo_output_dir = os.path.join(output_dir, "bayesian_optimization")
        optimizer = EvaluationBayesianOptimizer(
            max_trials=max_trials,
            n_test=n_test,
            device=device,
            output_dir=bo_output_dir,
            optimization_mode=optimization_mode
        )
        
        result = optimizer.run_optimization(checkpoint)
        
        if result is None:
            print("Optimization failed, running evaluation with default parameters...")
            # Fallback to default evaluation
            cmd = [
                "python", "eval_sim.py",
                "--checkpoint", checkpoint,
                "--output_dir", output_dir,
                "--device", device,
                "--n_test", str(n_test)
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
            
            subprocess.run(cmd, env=env)
        else:
            best_params, best_score = result
            
            print(f"\n{'='*60}")
            print(f"Running final evaluation with optimized parameters")
            print(f"Best params: {best_params}")
            print(f"Best score: {best_score:.4f}")
            print(f"{'='*60}\n")
            
            # Run final evaluation with best parameters
            cmd = [
                "python", "eval_sim.py",
                "--checkpoint", checkpoint,
                "--output_dir", output_dir,
                "--device", device,
                "--num_sampling_steps", str(best_params['num_sampling_steps']),
                "--cfg", str(best_params['cfg']),
                "--temperature", str(best_params['temperature']),
                "--n_test", str(n_test * 2)  # Use more tests for final evaluation
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
            
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                print("Final evaluation failed")
                return
            
            # Save best parameters to output directory
            best_params_file = os.path.join(output_dir, "best_params.json")
            with open(best_params_file, 'w') as f:
                json.dump({
                    'best_params': best_params,
                    'best_score': best_score
                }, f, indent=2)
            
            print(f"\nOptimization complete! Best parameters saved to {best_params_file}")


if __name__ == "__main__":
    main()
