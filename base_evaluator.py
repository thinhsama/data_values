import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils import check_random_state
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Union

from data_valuate import KNN_Shapley, DatasetDistance_geoloss, DatasetDistance_OT, KNNRegression, CKNN_Shapley, SAVA_OT, SAVA_OT_tanh, Hier_DatasetDistance_geoloss, Hier_DatasetDistance_OT  
from data_valuate import TMCSampler, ClassWiseShapley, BetaShapley, Hier_SAVA_OT
from visualize import plot_corrupted_sample_discovery
from experiment_method import discover_corrupted_sample, evaluate_label_noise, compute_WAD, evaluate_label_noise_20
# f1_score
from sklearn.metrics import f1_score
from machine_learning_model import ClassifierMLP, LogisticRegression
# time
import time
class BaseEvaluator:
    """
    Base class for different evaluation algorithms.
    Provides a common interface for evaluating data values.
    """
    def __init__(self, random_state: int = 42, device: torch.device = torch.device("cpu")):
        self.random_state = check_random_state(random_state)
        self.device = device
        torch.manual_seed(random_state)

    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Evaluate data values. Should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class KNNEvaluator(BaseEvaluator):
    """
    KNN-based evaluator using Shapley values.
    """
    def __init__(self, k_neighbors: int = 10, batch_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size

    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor) -> np.ndarray:
        dist_calculator = KNN_Shapley(x_train, y_train, x_valid, y_valid, k_neighbors=self.k_neighbors, batch_size=self.batch_size, random_state=self.random_state)
        dist_calculator.train_data_values()
        return dist_calculator.evaluate_data_values()
class CKNNEvaluator(BaseEvaluator):
    """
    KNN-based evaluator using Shapley values.
    """
    def __init__(self, k_neighbors: int = 10, T:int = 20, default:bool =1, batch_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors
        self.T = T
        self.batch_size = batch_size
        self.default = default
    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor) -> np.ndarray:
        dist_calculator = CKNN_Shapley(x_train, y_train, x_valid, y_valid, k_neighbors=self.k_neighbors, T= self.T, default=self.default ,batch_size=self.batch_size, random_state=self.random_state)
        dist_calculator.train_data_values()
        return dist_calculator.evaluate_data_values()

class LavaEvaluator_OT(BaseEvaluator):
    """
    Lava evaluator using Optimal Transport.
    """
    def __init__(self, lam_x: float = 1.0, lam_y: float = 1.0, ot_method: str = 'balance_ot_sinkhorn', **kwargs):
        super().__init__(**kwargs)
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.ot_method = ot_method

    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor) -> np.ndarray:
        dist_calculator = DatasetDistance_OT(x_train, y_train, x_valid, y_valid, device=self.device, lam_x=self.lam_x, lam_y=self.lam_y, ot_method=self.ot_method)
        u, _ = dist_calculator.dual_sol()
        return dist_calculator.compute_distance(u)
class LavaEvaluator_geomloss(BaseEvaluator):
    """
    Lava evaluator using Optimal Transport.
    """
    def __init__(self, lam_x: float = 1.0, lam_y: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lam_x = lam_x
        self.lam_y = lam_y

    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor) -> np.ndarray:
        dist_calculator = DatasetDistance_geoloss(x_train, y_train, x_valid, y_valid, device=self.device, lam_x=self.lam_x, lam_y=self.lam_y)
        u, _ = dist_calculator.dual_sol()
        return dist_calculator.compute_distance(u)
class HierLavaEvaluator(BaseEvaluator):
    """
    Lava evaluator using Optimal Transport.
    """
    def __init__(self, lam_x: float = 1.0, lam_y: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lam_x = lam_x
        self.lam_y = lam_y

    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor) -> np.ndarray:
        dist_calculator = Hier_DatasetDistance_OT(x_train, y_train, x_valid, y_valid, device=self.device, lam_x=self.lam_x, lam_y=self.lam_y)
        u, _ = dist_calculator.dual_sol()
        return dist_calculator.compute_distance(u)
class HierLavaEvaluator(BaseEvaluator):
    """
    Lava evaluator using Optimal Transport.
    """
    def __init__(self, lam_x: float = 1.0, lam_y: float = 1.0, batch = 32, **kwargs):
        super().__init__(**kwargs)
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.batch = batch
    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor) -> np.ndarray:
        dist_calculator = Hier_SAVA_OT(batch_size=self.batch)
        dist = dist_calculator.evaluate_data_values(x_train, y_train, x_valid, y_valid, lam_x=self.lam_x, lam_y=self.lam_y)
        return dist
class LavaEvaluator_batch(BaseEvaluator):
    """
    Lava evaluator using Optimal Transport.
    """
    def __init__(self, lam_x: float = 1.0, lam_y: float = 1.0, batch = 32, **kwargs):
        super().__init__(**kwargs)
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.batch = batch
    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor) -> np.ndarray:
        dist_calculator = SAVA_OT(batch_size=self.batch)
        dist = dist_calculator.evaluate_data_values(x_train, y_train, x_valid, y_valid, lam_x=self.lam_x, lam_y=self.lam_y)
        return dist
class TMCEvaluator:
    def __init__(self, model = None, mc_epochs: int = 100, min_cardinality: int = 5, **kwargs):
        self.model = model
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.sampler = TMCSampler(mc_epochs=mc_epochs, min_cardinality=min_cardinality, **kwargs)

    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor):
        shapley = ClassWiseShapley(self.sampler, self.model)
        shapley.input_data(x_train, y_train, x_valid, y_valid)
        shapley.train_data_values()
        return shapley.evaluate_data_values()
class BetaEvaluator:
    """
    Evaluator using Beta Shapley for data valuation.
    """
    def __init__(self, model=None ,alpha: int = 4, beta: int = 1, mc_epochs: int = 50, min_cardinality: int = 5, random_state: int = 42):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.random_state = random_state
        self.sampler = TMCSampler(mc_epochs=self.mc_epochs, min_cardinality=self.min_cardinality, random_state=self.random_state)
        self.beta_shapley = BetaShapley(sampler=self.sampler, model = self.model, alpha=self.alpha, beta=self.beta)
    def evaluate_data_values(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor) -> np.ndarray:
        """
        Evaluate data values using Beta Shapley.
        """
        self.beta_shapley.input_data(x_train, y_train, x_valid, y_valid)
        self.beta_shapley.train_data_values()
        return self.beta_shapley.evaluate_data_values()
class ExperimentRunner:
    """
    Orchestrates the execution of multiple evaluators on a dataset.
    """
    def __init__(self, evaluators: List[BaseEvaluator], output_dir: Optional[str] = None):
        self.evaluators = evaluators
        self.results = {}
        self.timings = {}
        self.output_dir = output_dir
        
        if output_dir:
            import pathlib
            self.output_dir = pathlib.Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor, **kwargs) -> Dict[str, np.ndarray]:
        """
        Run all evaluators on the dataset.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_valid
        self.y_test = y_valid
        i = 0
        for evaluator in self.evaluators:
            evaluator_name = self._get_evaluator_name(evaluator) + str(i)
            print(f"Running evaluator: {evaluator_name}")
            i+=1
            start_time = time.time()
            result = evaluator.evaluate_data_values(x_train, y_train, x_valid, y_valid, **kwargs)
            end_time = time.time()
            
            self.results[evaluator_name] = result
            self.timings[evaluator_name] = end_time - start_time
            print(f"{evaluator_name} completed in {end_time - start_time:.2f} seconds.")
        return self.results

    def _get_evaluator_name(self, evaluator):
        """
        Get the correct name of the evaluator (either function or class).
        """
        if hasattr(evaluator, '__name__'):
            return evaluator.__name__  # For functions
        elif hasattr(evaluator, '__class__'):
            return evaluator.__class__.__qualname__  # For class methods
        return str(evaluator)

    def save_results(self, file_name: str = "results.csv"):
        if not self.output_dir:
            print("Output directory is not set. Cannot save results.")
            return
        df = pd.DataFrame(self.results)
        save_path = self.output_dir / file_name
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    def plot_results(self, title: str = "Evaluation Results", save_plot: bool = False):
        if not self.results:
            print("No results to plot.")  
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        for evaluator_name, values in self.results.items():
            ax.plot(values, label=evaluator_name)

        ax.set_title(title)
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Values")
        ax.legend()
        plt.tight_layout()

        if save_plot and self.output_dir:
            plot_path = self.output_dir / "results_plot.png"
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        plt.show()

    def evaluate(self, noisy_train_indices: np.ndarray) -> Dict[str, Any]:
        evaluation_corrupt = {}
        i = 0
        for evaluator in self.evaluators:
            evaluator_name = self._get_evaluator_name(evaluator) + str(i)
            i+=1
            values = self.results[evaluator_name]
            evaluation_corrupt[evaluator_name] = discover_corrupted_sample(values, noisy_train_indices)
            
            #print(f"{evaluator_name}: {evaluation_corrupt[evaluator_name]}")
            plot_corrupted_sample_discovery(evaluation_corrupt[evaluator_name], evaluator_name=evaluator_name, noise_rate=0.2)
        
        return evaluation_corrupt
    def calculate_WAD(self, model, num_steps) -> Dict[str, Any]:
        WAD = {}
        i=0
        for evaluator in self.evaluators:
            evaluator_name = self._get_evaluator_name(evaluator) + str(i)
            i+=1
            values = self.results[evaluator_name]
            WAD[evaluator_name] = compute_WAD(model, self.x_train, self.y_train, self.x_test, self.y_test, values, num_steps)
            print(f"{evaluator_name}: {WAD[evaluator_name]}")
        return WAD
    def calculate_label_noise(self, noisy_train_indices: np.ndarray) -> Dict[str, Any]:
        label_noise = {}
        for evaluator in self.evaluators:
            evaluator_name = self._get_evaluator_name(evaluator)
            values = self.results[evaluator_name]
            label_noise[evaluator_name] = evaluate_label_noise(values, noisy_train_indices)
            print(f"{evaluator_name}: {label_noise[evaluator_name]}")
        return label_noise
    def calculate_label_noise_20(self, model, noisy_train_indices: np.ndarray, per:float=0.2) -> Dict[str, Any]:
        label_noise = {}
        i = 0
        for evaluator in self.evaluators:
            evaluator_name = self._get_evaluator_name(evaluator) + str(i)
            i+=1
            values = self.results[evaluator_name]
            label_noise[evaluator_name] = evaluate_label_noise_20(model, self.x_train, self.y_train, self.x_test, self.y_test, values, noisy_train_indices, per)
            print(f"{evaluator_name}: {label_noise[evaluator_name]}")
        return label_noise

# if __name__ == "__main__":
#     # Step 1: Prepare Dataset
#     num_train = 100
#     num_valid = 20
#     feature_dim = 10

#     x_train = torch.randn(num_train, feature_dim)
#     y_train = torch.randint(0, 2, (num_train,))
#     x_valid = torch.randn(num_valid, feature_dim)
#     y_valid = torch.randint(0, 2, (num_valid,))

#     # Step 2: Initialize Model
#     model = ClassifierMLP(input_dim=feature_dim, num_classes=2, layers=3, hidden_dim=25)

#     # Step 3: Initialize Sampler and Evaluator
#     css_evaluator = TMCEvaluator(model, mc_epochs=50, min_cardinality=5, random_state=42)
#     beta_evaluator = BetaEvaluator(model=model, alpha=4, beta=1, mc_epochs=50, min_cardinality=5, random_state=42)

#     # Step 4: Run Experiment
#     experiment = ExperimentRunner(evaluators=[beta_evaluator, css_evaluator])
#     results = experiment.run(x_train, y_train, x_valid, y_valid)

#     # Step 5: Output Results
#     print("Beta Shapley Data Values:")
#     print(results["BetaEvaluator"])
#     print("TMC-Shapley Data Values:")
#     print(results["TMCEvaluator"])
# if __name__ == "__main__":
#     # Step 1: Prepare Dataset
#     num_train = 100
#     feature_dim = 10
#     num_valid = 20
#     feature_dim = 10
#     x_train = torch.randn(num_train, feature_dim)
#     y_train = torch.randint(0, 2, (num_train,))
#     x_valid = torch.randn(num_valid, feature_dim)
#     y_valid = torch.randint(0, 2, (num_valid,))
#     # Step 2: Define Utility Function
#     def utility_func(subset_idx):
#         subset = x_train[subset_idx]  # Get the subset of the data
#         return subset.sum().item()  # Return the sum of the subset as utility

#     # Step 3: Initialize BetaEvaluator
#     beta_evaluator = BetaEvaluator(alpha=4, beta=1, mc_epochs=50, min_cardinality=5, random_state=42)
#     experiment = ExperimentRunner(evaluators=[beta_evaluator])
#     results = experiment.run(x_train, y_train, x_valid, y_valid, utility_func=utility_func)

#     #Step 4: Output Results
#     print("beta-Shapley Data Values:")
#     print(results['BetaEvaluator'])
# # if __name__ == "__main__":
#     # Step 1: Prepare Dataset
#     num_train = 100
#     num_valid = 20
#     feature_dim = 10

#     x_train = torch.randn(num_train, feature_dim)
#     y_train = torch.randint(0, 2, (num_train,))
#     x_valid = torch.randn(num_valid, feature_dim)
#     y_valid = torch.randint(0, 2, (num_valid,))

#     # Step 2: Initialize Evaluator
#     tmc_evaluator = TMCEvaluator(mc_epochs=50, min_cardinality=5, random_state=42)

#     # Step 3: Run Experiment
#     experiment = ExperimentRunner(evaluators=[tmc_evaluator])
#     results = experiment.run(x_train, y_train, x_valid, y_valid)

#     # Step 4: Output Results
#     print("TMC-Shapley Data Values:")
#     print(results['TMCEvaluator'])



# # Example usage
# if __name__ == "__main__":
#     # Dummy dataset
#     x_train = torch.rand(100, 10)
#     y_train = torch.randint(0, 2, (100,))
#     x_valid = torch.rand(20, 10)
#     y_valid = torch.randint(0, 2, (20,))
#     noisy_train_indices = np.random.choice(100, 20, replace=False)
#     print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, noisy_train_indices.shape)
#     # Instantiate evaluators
#     knn_evaluator = KNNEvaluator(k_neighbors=5, batch_size=16)
#     lava_evaluator = LavaEvaluator(lam_x=1.0, lam_y=1.0, ot_method="balance_ot_sinkhorn")

#     # Run experiment
#     experiment = ExperimentRunner(evaluators=[knn_evaluator, lava_evaluator], output_dir="./experiment_results")
#     results = experiment.run(x_train, y_train, x_valid, y_valid)

#     # Save results
#     experiment.save_results()

#     # Plot results
#     #experiment.plot_results(save_plot=True)

#     # Evaluate with a custom metric (e.g., mean of values)
#     def mean_metric(values: np.ndarray) -> float:
#         return np.mean(values)

#     scores = experiment.evaluate(noisy_train_indices)
#     print("Evaluation scores:", scores)