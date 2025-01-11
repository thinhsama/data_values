# experiment_method.py
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils import check_random_state
from typing import Dict, List

# Helper functions and models
from machine_learning_model import ClassifierMLP, LogisticRegression
from visualize import plot_performance, plot_corrupted_sample_discovery
from sklearn.metrics import accuracy_score, f1_score, auc, roc_auc_score
from tqdm import tqdm
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
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
def evaluate_label_noise(data_values:np.ndarray, noise_indices: np.ndarray)->dict[str, float]:
    #Indices with Values < 0
    #Found in Noisy Train Indices (TP)
    #False Positives (FP - Not Noisy but Selected)
    #False Negatives (FN - Noisy but Not Selected)
    # Tạo nhãn thực tế (1 là nhãn nhiễu, 0 là sạch)
    num_points = len(data_values)
    sorted_value_list = np.argsort(data_values)
    noise_pred_ind = data_values[sorted_value_list] < 0
    found_in_noisy = np.intersect1d(sorted_value_list[noise_pred_ind], noise_indices)
    not_in_noisy = np.setdiff1d(sorted_value_list[noise_pred_ind], noise_indices)
    not_found_in_small = np.setdiff1d(noise_indices, sorted_value_list[noise_pred_ind])
    TP = len(found_in_noisy)
    #print("found_in_noisy:", found_in_noisy)    
    FP = len(not_in_noisy)
    #print("not in noisy but selected:", not_in_noisy)
    FN = len(not_found_in_small)
    #print("noisy but not (selected or found in noisy):", not_found_in_small)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
def evaluate_label_noise_20(model, X_train, y_train, X_valid, y_valid , data_values: np.ndarray, noise_indices: np.ndarray, per: float=0.2) -> dict[str, float]:
    num_points = len(data_values)
    sorted_value_list = np.argsort(data_values)
    # Chọn ra 20% điểm tệ nhất (điểm có data_values thấp nhất)
    num_to_select = int(per * num_points)  # Lấy 20% trong tổng số điểm
    noise_pred_ind = sorted_value_list[:num_to_select]  # Chọn 20% điểm thấp nhất
    real_pred_ind = sorted_value_list[num_to_select:]  # Chọn 80% điểm còn lại
    model1 = model.clone()
    model1.fit(X_train[real_pred_ind], y_train[real_pred_ind], epochs=1000, lr = 0.01)
    y_pred = model1.predict(X_valid)
    F1_model = f1_score(y_valid, y_pred, average='macro')
    # Đếm TP, FP, FN
    found_in_noisy = np.intersect1d(noise_pred_ind, noise_indices)
    not_in_noisy = np.setdiff1d(noise_pred_ind, noise_indices)
    not_found_in_small = np.setdiff1d(noise_indices, noise_pred_ind)
    
    TP = len(found_in_noisy)
    FP = len(not_in_noisy)
    FN = len(not_found_in_small)
    
    # Tính Precision, Recall, F1-score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    #print("Found in noisy (TP):", found_in_noisy)
    #print("Not in noisy but selected (FP):", not_in_noisy)
    #print("Noisy but not selected (FN):", not_found_in_small)
    
    return {
        "F1-model": F1_model,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def compute_WAD(model, X_train, y_train, X_test, y_test, importance_order, num_steps=5):
    """
    Tính toán WAD (Weighted Average Drop) để đánh giá thứ tự mức độ quan trọng của các điểm dữ liệu, 
    với số lần xóa dữ liệu giới hạn.
    
    Args:
        model (LogisticRegression): Mô hình được huấn luyện.
        X_train (np.array): Dữ liệu huấn luyện.
        y_train (np.array): Nhãn huấn luyện.
        X_test (np.array): Dữ liệu kiểm thử.
        y_test (np.array): Nhãn kiểm thử.
        importance_order (list): Thứ tự mức độ quan trọng của các điểm dữ liệu.
        num_steps (int): Số lần đánh giá (bước nhảy) khi xóa dữ liệu. Mặc định là 50.
    
    Returns:
        float: Giá trị WAD.
    """
    n = len(importance_order)
    accuracy_drop = []
    model1 = model.clone()
    
    # Tính độ chính xác ban đầu với toàn bộ tập dữ liệu
    model1.fit(X_train, y_train)
    initial_accuracy = accuracy_score(y_test, model1.predict(X_test))

    # Chọn các bước xóa cách đều nhau
    sorted_importance_order = np.argsort(importance_order)[::-1]  # Sắp xếp thứ tự quan trọng
    steps = np.linspace(0, n, num_steps, dtype=int)  # Chọn 50 bước cách đều

    for step in tqdm(steps[1:], desc="Evaluating WAD"):  # Bỏ bước 0 vì không xóa điểm nào
        idx_to_keep = sorted_importance_order[step:]  # Chỉ giữ lại các điểm sau `step`
        if len(idx_to_keep) == 0:
            break
        model1 = model.clone()
        model1.fit(X_train[idx_to_keep], y_train[idx_to_keep], epochs=100, lr=0.1)
        new_accuracy = accuracy_score(y_test, model1.predict(X_test))
        drop = initial_accuracy - new_accuracy
        accuracy_drop.append(drop)

    # Tính WAD theo công thức
    wad = np.sum([1 / step * np.sum(accuracy_drop[:i]) for i, step in enumerate(steps[1:], start=1)])
    return wad


# # Ví dụ
# X_train = np.random.rand(100, 10)
# y_train = np.random.randint(0, 2, 100)
# X_test = np.random.rand(30, 10)
# y_test = np.random.randint(0, 2, 30)
# importance_order = np.arange(100)  # Ví dụ: Thứ tự giảm dần theo mức độ quan trọng

# model = LogisticRegression()
# wad_result = compute_WAD(model, X_train, y_train, X_test, y_test, importance_order)
# print(f"WAD: {wad_result}")

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