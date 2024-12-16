# experiment_method.py
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils import check_random_state
from typing import Dict, List

# Helper functions and models
from machine_learning_model import ClassifierMLP, LogisticRegression
from visualize import plot_performance, plot_corrupted_sample_discovery
def noisy_detection(data_values: np.ndarray, noise_train_indices: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the ability to identify noisy indices using F1 score.

    Parameters:
    - data_values: np.ndarray - Scores computed by a model evaluator.
    - noise_train_indices: np.ndarray - Indices of noisy data.

    Returns:
    - dict with F1 score.
    """
    sorted_indices = np.argsort(data_values)
    precision = len(np.intersect1d(sorted_indices[:len(noise_train_indices)], noise_train_indices)) / len(noise_train_indices)
    recall = len(np.intersect1d(sorted_indices[:len(noise_train_indices)], noise_train_indices)) / len(sorted_indices)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {'f1_score': f1}
def get_model(model_name, input_dim, num_classes):
    """Utility function to return model based on the model_name."""
    if model_name == 'LogisticRegression':
        return LogisticRegression(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'MLP':
        return ClassifierMLP(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

def remove_high_low(data_values: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
                    x_valid: np.ndarray, y_valid: np.ndarray, model_name: str, epochs:int = 100,
                    percentile: float = 0.05) -> dict[str, list[float]]:
    """
    Evaluate performance after removing high/low valued points determined by data values.
    """
    input_dim = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    num_points = len(x_train)
    num_period = max(round(num_points * percentile), 5)

    sorted_value_list = np.argsort(data_values)
    valuable_list, unvaluable_list, randomize_list = [], [], []

    rs = check_random_state(0)
    random_indices = rs.permutation(num_points)
    sorted_random_list = np.argsort(random_indices)

    for bin_index in range(0, num_points, num_period):
        # Valuable model: Keep most valuable
        most_values_idx = sorted_value_list[bin_index:]
        valuable_model = get_model(model_name, input_dim, num_classes)
        valuable_model.fit(torch.tensor(x_train[most_values_idx], dtype=torch.float32),
                           torch.tensor(y_train[most_values_idx], dtype=torch.long), epochs=epochs)
        y_pred = valuable_model.predict(torch.tensor(x_valid, dtype=torch.float32))
        valuable_list.append(f1_score(y_valid, y_pred, average='macro'))

        # Unvaluable model: Remove least valuable
        least_values_idx = sorted_value_list[:num_points - bin_index]
        unvaluable_model = get_model(model_name, input_dim, num_classes)
        unvaluable_model.fit(torch.tensor(x_train[least_values_idx], dtype=torch.float32),
                             torch.tensor(y_train[least_values_idx], dtype=torch.long), epochs=epochs)
        y_pred = unvaluable_model.predict(torch.tensor(x_valid, dtype=torch.float32))
        unvaluable_list.append(f1_score(y_valid, y_pred, average='macro'))

        # Randomized model
        random_values_idx = sorted_random_list[bin_index:]
        random_model = get_model(model_name, input_dim, num_classes)
        random_model.fit(torch.tensor(x_train[random_values_idx], dtype=torch.float32),
                         torch.tensor(y_train[random_values_idx], dtype=torch.long), epochs=epochs)
        y_pred = random_model.predict(torch.tensor(x_valid, dtype=torch.float32))
        randomize_list.append(f1_score(y_valid, y_pred, average='macro'))

    x_axis = [i / num_points for i in range(0, num_points, num_period)]

    return {
        'valuable_list': valuable_list,
        'unvaluable_list': unvaluable_list,
        'randomize_list': randomize_list,
        'axis': x_axis
    }

def discover_corrupted_sample(data_values: np.ndarray, noisy_train_indices: np.ndarray,
                              percentile: float = 0.05) -> dict[str, list[float]]:
    """
    Detect noisy indices among low data values.
    """
    num_points = len(data_values)
    num_period = max(round(num_points * percentile), 5)
    sorted_value_list = np.argsort(data_values, kind='stable')

    found_rates = []
    for bin_index in range(0, num_points + num_period, num_period):
        found_rates.append(len(np.intersect1d(sorted_value_list[:bin_index], noisy_train_indices)) /
                           len(noisy_train_indices))

    x_axis = [i / len(found_rates) for i in range(len(found_rates))]
    return {
        'found_rates': found_rates,
        'axis': x_axis
    }

def para_bin_remove(data_values: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
                    x_valid: np.ndarray, y_valid: np.ndarray, model_name: str,
                    bin_size: int = 1) -> dict[str, list[float]]:
    """
    Evaluate performance across bins for both valuable and unvaluable removals.
    """
    input_dim = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    num_points = len(x_train)

    bins_indices = [*range(5, num_points - 1, bin_size), num_points - 1]
    sorted_indices = np.argsort(data_values)
    random_indices = check_random_state(0).permutation(num_points)
    sorted_random_list = np.argsort(random_indices)

    frac_list = [(i + 1) / num_points for i in bins_indices]
    perf_values, perf_unvalues, random_values = [], [], []

    for bin_index in bins_indices:
        coalition = sorted_indices[bin_index:]
        model_tmp = get_model(model_name, input_dim, num_classes)
        model_tmp.fit(torch.tensor(x_train[coalition], dtype=torch.float32),
                      torch.tensor(y_train[coalition], dtype=torch.long), epochs=10)
        y_pred = model_tmp.predict(torch.tensor(x_valid, dtype=torch.float32))
        perf_values.append(f1_score(y_valid, y_pred, average='macro'))

        coalition = sorted_indices[:num_points - bin_index]
        model_tmp = get_model(model_name, input_dim, num_classes)
        model_tmp.fit(torch.tensor(x_train[coalition], dtype=torch.float32),
                      torch.tensor(y_train[coalition], dtype=torch.long), epochs=10)
        y_pred = model_tmp.predict(torch.tensor(x_valid, dtype=torch.float32))
        perf_unvalues.append(f1_score(y_valid, y_pred, average='macro'))

        coalition = sorted_random_list[bin_index:]
        model_tmp = get_model(model_name, input_dim, num_classes)
        model_tmp.fit(torch.tensor(x_train[coalition], dtype=torch.float32),
                      torch.tensor(y_train[coalition], dtype=torch.long), epochs=10)
        y_pred = model_tmp.predict(torch.tensor(x_valid, dtype=torch.float32))
        random_values.append(f1_score(y_valid, y_pred, average='macro'))

    return {
        'frac_datapoints_explored': frac_list,
        'values': perf_values,
        'unvalues': perf_unvalues,
        'random_values': random_values
    }
def cut_top_down(data_values: np.ndarray, x_train, y_train, x_valid, y_valid, model_name: str) -> dict[str, float]:
    """
    Split data into top and bottom halves based on data values and evaluate performance.

    Parameters:
        data_values: np.ndarray - Data values for sorting.
        x_train, y_train: Training data and labels.
        x_valid, y_valid: Validation data and labels.
        model_name: str - Name of the model ('LogisticRegression' or 'MLP').

    Returns:
        Dictionary with F1 scores for the top (valuable) and bottom (unvaluable) halves.
    """
    input_dim = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    half_size = len(x_train) // 2

    sorted_value_list = np.argsort(data_values)
    top_half_indices = sorted_value_list[half_size:]
    bottom_half_indices = sorted_value_list[:half_size]

    # Train on top half
    valuable_model = get_model(model_name, input_dim=input_dim, num_classes=num_classes)
    valuable_model.fit(torch.tensor(x_train[top_half_indices], dtype=torch.float32),
                       torch.tensor(y_train[top_half_indices], dtype=torch.long), epochs=1000)
    y_pred_valuable = valuable_model.predict(torch.tensor(x_valid, dtype=torch.float32))
    valuable_f1 = f1_score(y_valid, y_pred_valuable, average='macro')

    # Train on bottom half
    unvaluable_model = get_model(model_name, input_dim=input_dim, num_classes=num_classes)
    unvaluable_model.fit(torch.tensor(x_train[bottom_half_indices], dtype=torch.float32),
                         torch.tensor(y_train[bottom_half_indices], dtype=torch.long), epochs=1000)
    y_pred_unvaluable = unvaluable_model.predict(torch.tensor(x_valid, dtype=torch.float32))
    unvaluable_f1 = f1_score(y_valid, y_pred_unvaluable, average='macro')

    return {
        'valuable_f1': valuable_f1,
        'unvaluable_f1': unvaluable_f1,
    }
def performance_remove_noise(data_values: np.ndarray, noisy_train_indices: np.ndarray, x_train, y_train,
                             x_valid, y_valid, model_name: str, percentile: float = 0.05) -> dict[str, float]:
    """
    Evaluate model performance by progressively removing noisy samples.

    Parameters:
        data_values: np.ndarray - Data values used for sorting.
        noisy_train_indices: np.ndarray - Indices of noisy training samples.
        x_train, y_train: Training data and labels.
        x_valid, y_valid: Validation data and labels.
        model_name: str - Name of the model ('LogisticRegression' or 'MLP').
        percentile: float - Fraction of data points to remove in each step.

    Returns:
        Dictionary with F1 scores for various noise removal strategies.
    """
    input_dim = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    num_points = len(x_train)
    num_period = max(round(num_points * percentile), 5)

    # Helper function for fitting and evaluating a model
    def fit_and_evaluate(indices):
        temp_model = get_model(model_name, input_dim=input_dim, num_classes=num_classes)
        temp_model.fit(torch.tensor(x_train[indices], dtype=torch.float32),
                       torch.tensor(y_train[indices], dtype=torch.long), epochs=1000)
        y_pred = temp_model.predict(torch.tensor(x_valid, dtype=torch.float32))
        return f1_score(y_valid, y_pred, average='macro')

    # Evaluate different noise removal scenarios
    f1_scores = {
        'remove_noise_f1': fit_and_evaluate(np.delete(np.arange(num_points), noisy_train_indices[:num_period])),
        'remove_noise1_f1': fit_and_evaluate(np.delete(np.arange(num_points), noisy_train_indices[:2 * num_period])),
        'remove_noise2_f1': fit_and_evaluate(np.delete(np.arange(num_points), noisy_train_indices[:3 * num_period])),
        'remove_bnoise_f1': fit_and_evaluate(np.delete(np.arange(num_points), noisy_train_indices[-num_period:])),
        'remove_bnoise1_f1': fit_and_evaluate(np.delete(np.arange(num_points), noisy_train_indices[-2 * num_period:])),
    }

    return f1_scores
# if __name__ == "__main__":
#     # Dữ liệu giả lập
#     num_samples = 100
#     input_dim = 10
#     num_classes = 3
#     data_values = np.random.rand(num_samples)
#     x_train = np.random.rand(num_samples, input_dim)
#     y_train = np.random.randint(0, num_classes, num_samples)
#     x_valid = np.random.rand(20, input_dim)
#     y_valid = np.random.randint(0, num_classes, 20)
#     noise_indices = np.random.choice(num_samples, size=10, replace=False)

#     # Noisy detection
#     #print("\nNoisy Detection:")
#     #noisy_result = noisy_detection(data_values, noise_indices)
#     #print("Noisy Detection Result:", noisy_result)

#     # Remove high/low value samples
#     print("\nRemoving High/Low Value Samples:")
#     result_high_low = remove_high_low(data_values, x_train, y_train, x_valid, y_valid, model_name='MLP', epochs=50)
#     print("High/Low Removal Result:", result_high_low)

#     # Discover corrupted samples
#     print("\nDiscovering Corrupted Samples:")
#     corrupted_result = discover_corrupted_sample(data_values, noise_indices)
#     print("Corrupted Sample Discovery Result:", corrupted_result)

#     # Performance by removing noise
#     print("\nPerformance by Removing Noise:")
#     performance_result = performance_remove_noise(data_values, noise_indices, x_train, y_train, x_valid, y_valid, model_name='LogisticRegression', percentile=0.1)
#     print("Performance Result:", performance_result)
#     # Visualize results
#     plot_performance(result_high_low, evaluator_name='MLP')
#     plot_corrupted_sample_discovery(corrupted_result, evaluator_name='MLP', noise_rate=0.1)