# nosify.py
import numpy as np
from numpy.random import RandomState
from typing import Dict, Optional  # Bổ sung để định kiểu dữ liệu trả về

def mix_label(y_train: np.ndarray, y_valid: np.ndarray = None, noise_rate: float = 0.2, random_state: int = 0) -> Dict[str, np.ndarray]:
    """
    Hàm trộn nhãn (label) bằng cách thêm nhiễu ngẫu nhiên vào dữ liệu huấn luyện và kiểm tra.

    Parameters:
    - y_train: Dữ liệu nhãn huấn luyện (numpy array).
    - y_valid: Dữ liệu nhãn kiểm tra (numpy array).
    - noise_rate: Tỷ lệ nhiễu được thêm vào.
    - random_state: Giá trị random seed để tái tạo kết quả.

    Returns:
    - dict chứa:
        - y_train: Nhãn huấn luyện sau khi thêm nhiễu.
        - y_valid: Nhãn kiểm tra sau khi thêm nhiễu.
        - noisy_train_indices: Các chỉ số của nhãn huấn luyện bị nhiễu.
    """
    rs = RandomState(random_state)
    num_train = len(y_train)
    if y_valid is None:
        num_valid = 0
    else:
        num_valid = len(y_valid)
    print(num_valid)
    num_noisy_t = int(noise_rate * num_train)
    num_noisy_v = int(noise_rate * num_valid)

    train_replace = rs.choice(num_train, num_noisy_t, replace=False)
    valid_replace = rs.choice(num_valid, num_noisy_v, replace=False)

    train_classes, train_mapping = np.unique(y_train, return_inverse=True)
    valid_classes, valid_mapping = np.unique(y_valid, return_inverse=True)

    train_shift = rs.choice(len(train_classes) - 1, len(train_replace)) + 1
    valid_shift = rs.choice(len(valid_classes) - 1, len(valid_replace)) + 1

    y_train[train_replace] = train_classes[(train_mapping[train_replace] + train_shift) % len(train_classes)]
    y_valid[valid_replace] = valid_classes[(valid_mapping[valid_replace] + valid_shift) % len(valid_classes)]

    return {
        'y_train': y_train,
        'y_valid': y_valid,
        'noisy_train_indices': train_replace,
    }

def add_gauss_noise(x_train: np.ndarray, x_valid: np.ndarray, x_test: Optional[np.ndarray] = None, noise_rate: float = 0.2, mu: float = 0.0, sigma: float = 1.0, random_state: int = 0) -> Dict[str, np.ndarray]:
    """
    Hàm thêm nhiễu Gaussian vào dữ liệu đầu vào.

    Parameters:
    - x_train: Dữ liệu huấn luyện (numpy array).
    - x_valid: Dữ liệu kiểm tra (numpy array).
    - x_test: Dữ liệu kiểm thử (nếu có, numpy array).
    - noise_rate: Tỷ lệ nhiễu được thêm vào.
    - mu: Trung bình của phân phối Gaussian.
    - sigma: Độ lệch chuẩn của phân phối Gaussian.
    - random_state: Giá trị random seed để tái tạo kết quả.

    Returns:
    - dict chứa:
        - x_train: Dữ liệu huấn luyện sau khi thêm nhiễu.
        - x_valid: Dữ liệu kiểm tra sau khi thêm nhiễu.
        - x_test: Dữ liệu kiểm thử sau khi thêm nhiễu (nếu có).
        - noisy_train_indices: Các chỉ số của dữ liệu huấn luyện bị nhiễu.
    """
    rs = np.random.RandomState(random_state)
    num_train = len(x_train)
    num_valid = len(x_valid)
    num_noisy_t = int(noise_rate * num_train)
    num_noisy_v = int(noise_rate * num_valid)

    noise_train_indices = rs.choice(num_train, num_noisy_t, replace=False)
    noise_valid_indices = rs.choice(num_valid, num_noisy_v, replace=False)

    noise_train = rs.normal(mu, sigma, size=(num_noisy_t, x_train.shape[1])).astype(np.float32)
    noise_valid = rs.normal(mu, sigma, size=(num_noisy_v, x_valid.shape[1])).astype(np.float32)

    x_train[noise_train_indices] += noise_train
    x_valid[noise_valid_indices] += noise_valid

    result = {
        'x_train': x_train,
        'x_valid': x_valid,
        'noisy_train_indices': noise_train_indices,
    }

    if x_test is not None:
        num_test = len(x_test)
        num_noisy_test = int(noise_rate * num_test)
        noise_test_indices = rs.choice(num_test, num_noisy_test, replace=False)
        noise_test = rs.normal(mu, sigma, size=(num_noisy_test, x_test.shape[1]))
        x_test[noise_test_indices] += noise_test
        result['x_test'] = x_test

    return result

# if __name__ == "__main__":
#     # Dữ liệu giả lập
#     y_train = np.array([0, 1, 1, 0, 1, 0])
#     y_valid = np.array([1, 0, 0, 1])
#     x_train = np.random.rand(6, 2)
#     x_valid = np.random.rand(4, 2)
#     x_test = np.random.rand(3, 2)

#     # Gọi hàm mix_label
#     mixed_labels = mix_label(y_train, y_valid, noise_rate=0.3, random_state=42)
#     print("Mixed Labels:", mixed_labels)

#     # Gọi hàm add_gauss_noise
#     noisy_data = add_gauss_noise(x_train, x_valid, x_test, noise_rate=0.3, mu=0.0, sigma=0.1, random_state=42)
#     print("Noisy Data:", noisy_data)
