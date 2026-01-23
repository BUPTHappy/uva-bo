import optuna
import json
from typing import Dict, Any, Tuple

class UVABayesianOptimizer:
    """
    Bayesian optimizer for Unified Video Action (UVA) model.
    Optimizes parameters that exist in both DGM and UVA:
    - num_sampling_steps
    - cfg
    - temperature
    """
    def __init__(self, max_trials: int = 30, optimization_mode: str = "balanced"):
        self.max_trials = max_trials
        self.optimization_mode = optimization_mode
        self.study = optuna.create_study(direction='maximize')
        self.best_params = None
        
        # Validate optimization mode
        valid_modes = ["speed_priority", "performance_priority", "balanced"]
        if optimization_mode not in valid_modes:
            raise ValueError(f"Invalid optimization_mode: {optimization_mode}. Must be one of {valid_modes}")

    def optimize_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define parameter ranges based on optimization mode.
        Only includes parameters that exist in UVA.
        """
        if self.optimization_mode == "speed_priority":
            # Fewer sampling steps for speed
            num_sampling_steps_options = [1, 2, 3, 4, 5, 10, 20, 30, 50]
            temperature_range = (0.5, 1.0)  # Lower temperature for faster convergence
            cfg_range = (0.5, 1.5)  # Narrower CFG range
            
        elif self.optimization_mode == "performance_priority":
            # More sampling steps for performance
            num_sampling_steps_options = [10, 20, 30, 50, 75, 100, 150, 200]
            temperature_range = (0.5, 1.5)  # Wider temperature range
            cfg_range = (0.5, 2.0)  # Wider CFG range
            
        else:  # balanced
            # Balanced options
            num_sampling_steps_options = [5, 10, 20, 30, 50, 75, 100]
            temperature_range = (0.5, 1.5)
            cfg_range = (0.5, 1.5)
        
        params = {
            'num_sampling_steps': trial.suggest_categorical('num_sampling_steps', num_sampling_steps_options),
            'cfg': trial.suggest_float('cfg', *cfg_range),
            'temperature': trial.suggest_float('temperature', *temperature_range),
        }
        
        return params
    
    def optimize(self, objective_func) -> Tuple[Dict[str, Any], float]: 
        """
        Run Bayesian optimization.
        
        Args:
            objective_func: Function that takes params dict and returns score
            
        Returns:
            Tuple of (best_params, best_score)
        """
        def objective(trial):
            params = self.optimize_params(trial) 
            score = objective_func(params)
            if self.best_params is None or score > self.best_params[1]:
                self.best_params = (params, score)
            return score
        
        self.study.optimize(objective, n_trials=self.max_trials)
        return self.best_params 
    
    def save_results(self, filename: str = "optimization_results.json"):
        """Save optimization results to JSON file."""
        if self.best_params:
            results = {
                'best_params': self.best_params[0],
                'best_score': self.best_params[1],
                'all_trials': [
                    {'params': trial.params, 'value': trial.value}
                    for trial in self.study.trials
                ]
            }
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
