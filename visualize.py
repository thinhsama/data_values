# plot detect noise
import matplotlib.pyplot as plt

# def plot_corrupted_sample_discovery(corrupted_sample_param, evaluator_name='LAVA', noise_rate=0.2):
#     """
#     Hàm vẽ biểu đồ kết quả phát hiện nhãn bị lỗi dựa trên các giá trị đã cho.

#     Parameters:
#     - corrupted_sample_param: dict chứa các thông tin 'axis' và 'found_rates_low'.
#     - evaluator_name: Tên của evaluator sẽ hiển thị trong tiêu đề (mặc định là 'LAVA').
#     - noise_rate: Tỷ lệ nhiễu trong dữ liệu (mặc định là 0.2).
#     """
#     num_bins = len(corrupted_sample_param['axis'])
#     x_axis = corrupted_sample_param['axis']
#     found_rates = corrupted_sample_param['found_rates']

#     # Corrupted label discovery results (dvrl, optimal, random)
#     y_dv = found_rates[:num_bins]
#     y_opt = [min((i / num_bins / noise_rate, 1.0)) for i in range(len(found_rates))]
#     y_random = x_axis

#     # Tạo biểu đồ
#     plt.figure(figsize=(10, 6))
#     plt.plot(x_axis, y_dv, "o-", label="Evaluator")
#     plt.plot(x_axis, y_opt, "--", label="Optimal")
#     plt.plot(x_axis, y_random, ":", label="Random")

#     # Thiết lập nhãn cho trục
#     plt.xlabel("Proportion of data inspected")
#     plt.ylabel("Proportion of discovered corrupted samples")

#     # Thêm chú thích
#     plt.legend()

#     # Đặt tiêu đề dựa trên tên của evaluator
#     plt.title(f'Evaluator: {evaluator_name}')

#     # Hiển thị biểu đồ
#     plt.grid(True)
#     plt.show()
def plot_corrupted_sample_discovery(
    corrupted_sample_param, 
    evaluator_name='LAVA', 
    noise_rate=0.2,
    is_new_fig=True
):
    """
    Hàm vẽ biểu đồ kết quả phát hiện nhãn bị lỗi dựa trên các giá trị đã cho.
    Nếu is_new_fig=True, sẽ tạo figure mới mỗi lần; nếu False, vẽ vào figure đang có.
    """
    # Nếu muốn vẽ nhiều evaluator chung 1 plot, ta KHÔNG tạo figure mới ở đây
    if is_new_fig:
        plt.figure(figsize=(10, 6))

    num_bins = len(corrupted_sample_param['axis'])
    x_axis = corrupted_sample_param['axis']
    found_rates = corrupted_sample_param['found_rates']

    # Tính y_opt, y_random
    y_dv = found_rates[:num_bins]
    y_opt = [min((i / num_bins / noise_rate, 1.0)) for i in range(len(found_rates))]
    y_random = x_axis

    plt.plot(x_axis, y_dv, "o-", label=f"Evaluator: {evaluator_name}")
    plt.plot(x_axis, y_opt, "--", label=f"Optimal ({evaluator_name})")
    plt.plot(x_axis, y_random, ":", label=f"Random ({evaluator_name})")

    if is_new_fig:
        plt.xlabel("Proportion of data inspected")
        plt.ylabel("Proportion of discovered corrupted samples")
        plt.title(f'Evaluator: {evaluator_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
# plot_corrupted_sample_discovery(corrupted_sample_param, evaluator_name='LAVA', noise_rate=0.2)
def plot_performance(evaluation_param, evaluator_name='LAVA'):
    # Lấy dữ liệu từ evaluation_param
    x_axis = evaluation_param['axis']
    valuable_list = evaluation_param['valuable_list']
    unvaluable_list = evaluation_param['unvaluable_list']
    randomize_list = evaluation_param['randomize_list']

    # Tạo biểu đồ
    plt.figure(figsize=(10, 6))

    # Vẽ các đường biểu diễn
    plt.plot(x_axis, valuable_list, marker="o", linestyle="-", label="Removing low value data")
    plt.plot(x_axis, unvaluable_list, marker="x", linestyle="-", label="Removing high value data")
    plt.plot(x_axis, randomize_list, marker="^", linestyle="-", label="Randomly removing data")

    # Thiết lập nhãn cho các trục
    plt.xlabel("Fraction Removed")
    plt.ylabel('Metric: F1')

    # Thêm chú thích
    plt.legend()

    # Đặt tiêu đề dựa trên tên của evaluator
    plt.title(f"Performance: {evaluator_name}")

    # Hiển thị biểu đồ
    plt.grid(True)
    plt.show()