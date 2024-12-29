import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
import pickle
import torch
# Data Experiment

def data_experiment(shap_vals_lst, shap_vals_algo_lst, X_train, y_train, X_test, y_test,
                    model, metrics, plot_every_percentage,
                    add_data=False, low_value_first=True):
    
    frac_data = []
    sort_val_idxs_lst = []

    num_trn = len(X_train)

    # Sort SHAP values and indices
    for shap_vals in shap_vals_lst:
        sorted_indices = np.argsort(shap_vals)
        if not low_value_first:
            sorted_indices = sorted_indices[::-1]  # Reverse if removing high values
        sort_val_idxs_lst.append(sorted_indices.copy())

    acc_score_lst = [[] for _ in range(len(shap_vals_lst))]

    # Loop through the data to remove or add incrementally
    for i in range(0, num_trn, round(num_trn * plot_every_percentage)):
        frac_data.append(round(i / num_trn, 2) * 100)

        for j, sorted_idxs in enumerate(sort_val_idxs_lst):
            if add_data:
                selected_idxs = sorted_idxs[:i]
            else:
                selected_idxs = sorted_idxs[i:]
            
            if len(selected_idxs) == 0:
                acc = 0  # Assign 0 if no data is selected
            else:
                model1 = model.clone()
                X_train_subset = X_train.index_select(0, torch.tensor(selected_idxs, dtype=torch.long)).detach().clone()
                y_train_subset = y_train.index_select(0, torch.tensor(selected_idxs, dtype=torch.long)).detach().clone()
                model1.fit(X_train_subset, y_train_subset, epochs=100, lr=0.1)
                preds = model1.predict(X_test)
                #acc = f1_score(y_test, preds, average='weighted') if metrics == 'acc' else roc_auc_score(y_test, preds)
                acc = accuracy_score(y_test, preds) if metrics == 'acc' else roc_auc_score(y_test, preds)
            acc_score_lst[j].append(acc)

    data_dict = dict(zip(shap_vals_algo_lst, acc_score_lst))
    data_key = 'frac_data_added' if add_data else 'frac_data_removed'
    data_dict[data_key] = frac_data
    # Plot results
    for algo_name in shap_vals_algo_lst:
        plt.plot(data_dict[data_key], data_dict[algo_name], label=algo_name)
    plt.legend(loc='best')
    plt.show()
    return data_dict


# Run Multiple Experiments

def run_experiments(shap_vals_lst, shap_vals_algo_lst, X_train, y_train, X_test, y_test, 
                    model, metrics='acc'):
    
    experiment_configs = [
        (False, True, 'remove_low'),
        (False, False, 'remove_high'),
        (True, True, 'add_low'),
        (True, False, 'add_high')
    ]

    all_results = {}
    for add_data, low_value_first, suffix in experiment_configs:
        print(f'Running {suffix} experiment...')
        result = data_experiment(
            shap_vals_lst, shap_vals_algo_lst, X_train, y_train, X_test, y_test,
            model, metrics, plot_every_percentage=0.05,
            add_data=add_data, low_value_first=low_value_first
        )
        all_results[suffix] = result

    # Plot final results after all experiments
    final_plot_avg(all_results, shap_vals_algo_lst, remove_add_ratio=30, xticks=list(range(10, 31, 10)), imbalance=False)


# Final Plotting Function

# def final_plot_avg(data_dict, shap_algo_lst, remove_add_ratio, xticks, imbalance=False):
#     fig, axes = plt.subplots(1, 4, figsize=(15, 2))
#     title_font, label_font, ticks_font = 10, 8, 8

#     color_map = {'knn': 'orange', 'lava': 'green', 'random': 'grey'}
#     line_style_map = {'knn': 'solid', 'lava': 'solid', 'random': 'solid'}
#     marker_map = {'knn': '+', 'lava': 'o', 'random': 'd'}

#     data_lst = [data_dict['remove_low'], data_dict['remove_high'], data_dict['add_low'], data_dict['add_high']]
#     keys = ['frac_data_removed', 'frac_data_removed', 'frac_data_added', 'frac_data_added']

#     subtitles = ['Removing low value data', 'Removing high value data', 'Adding low value data', 'Adding high value data']
#     x_labels = ['Fraction of data removed (%)'] * 2 + ['Fraction of data added (%)'] * 2

#     for i in range(4):
#         ymin, ymax = 1, 0
#         for j in range(len(shap_algo_lst)):
#             x_vals = np.array(data_lst[i][keys[i]])
#             bool_ratio = x_vals <= remove_add_ratio

#             results = np.array(data_lst[i][shap_algo_lst[j]])
#             normalized_results = results / np.clip(results[0], 1e-8, None)  # Prevent division by zero
#             mean_rslts = np.mean(normalized_results)
#             confidence_band = 1.645 * np.std(normalized_results) / np.sqrt(len(normalized_results))

#             axes[i].plot(
#                 x_vals[bool_ratio],
#                 np.full_like(x_vals[bool_ratio], mean_rslts),
#                 color=color_map[shap_algo_lst[j]],
#                 linestyle=line_style_map[shap_algo_lst[j]],
#                 marker=marker_map[shap_algo_lst[j]],
#                 lw=2, markersize=4
#             )
#             axes[i].fill_between(
#                 x_vals[bool_ratio],
#                 mean_rslts - confidence_band,
#                 mean_rslts + confidence_band,
#                 color=color_map[shap_algo_lst[j]],
#                 alpha=0.1
#             )

#         axes[i].set_xticks(xticks)
#         axes[i].set_xlabel(x_labels[i], fontsize=label_font)
#         axes[i].set_ylabel('Relative accuracy (%)', fontsize=label_font)
#         axes[i].set_title(subtitles[i], fontsize=title_font, fontweight='bold')

#     plt.subplots_adjust(left=0.1, right=0.85, wspace=0.5)
#     plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def final_plot_avg(data_dict, shap_algo_lst, remove_add_ratio, xticks, imbalance=False):
    """
    Vẽ biểu đồ kết quả thí nghiệm loại bỏ/thêm dữ liệu cho nhiều thuật toán.

    Args:
        data_dict: Dictionary chứa kết quả thí nghiệm.
        shap_algo_lst: Danh sách tên của các thuật toán SHAP.
        remove_add_ratio: Ngưỡng phần trăm loại bỏ/thêm dữ liệu.
        xticks: Danh sách nhãn cho trục x.
        imbalance: Biến cho biết dữ liệu có bị mất cân bằng hay không.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 2))
    title_font, label_font, ticks_font = 10, 8, 8

    # Định nghĩa màu sắc, kiểu đường và marker
    color_map = {algo: plt.cm.get_cmap('viridis')(i / (len(shap_algo_lst) - 1)) if len(shap_algo_lst) > 1 else 'blue' for i, algo in enumerate(shap_algo_lst)}
    line_style_map = {algo: 'solid' for algo in shap_algo_lst}
    marker_map = {algo: ['*', '+', 'o', 'd'][i % 4] for i, algo in enumerate(shap_algo_lst)}

    data_lst = [data_dict['remove_low'], data_dict['remove_high'], data_dict['add_low'], data_dict['add_high']]
    keys = ['frac_data_removed', 'frac_data_removed', 'frac_data_added', 'frac_data_added']

    subtitles = ['Removing low value data', 'Removing high value data',
                 'Adding low value data', 'Adding high value data']
    x_labels = ['Fraction of data removed (%)'] * 2 + ['Fraction of data added (%)'] * 2

    for i in range(4):
        y_min, y_max = 1, 0
        x_vals = np.array(data_lst[i][keys[i]])
        bool_ratio = x_vals <= remove_add_ratio

        for j, shap_algo in enumerate(shap_algo_lst):
            results = np.array(data_lst[i][shap_algo])
            normalized_results = results / np.clip(results[0], 1e-8, None)

            axes[i].plot(
                x_vals[bool_ratio],
                normalized_results[bool_ratio],
                color=color_map[shap_algo],
                linestyle=line_style_map[shap_algo],
                marker=marker_map[shap_algo],
                lw=2, markersize=4, label=shap_algo
            )
            if min(normalized_results[bool_ratio]) < y_min:
                y_min = min(normalized_results[bool_ratio])
            if max(normalized_results[bool_ratio]) > y_max:
                y_max = max(normalized_results[bool_ratio])

        axes[i].set_xticks(xticks)
        axes[i].set_ylim((y_min-0.03, y_max+0.01) if i == 0 else (y_min-0.01, y_max+0.01))
        axes[i].set_xlabel(x_labels[i], fontsize=label_font)
        axes[i].set_ylabel('Relative accuracy (%)' if not imbalance else 'Relative AUC (%)', fontsize=label_font)
        axes[i].set_title(subtitles[i], fontsize=title_font, fontweight='bold')

    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.5)

    # colorbar
    cmap = ListedColormap([color_map[algo] for algo in shap_algo_lst])
    norm = mpl.colors.BoundaryNorm(boundaries=np.arange(len(shap_algo_lst) + 1) - 0.5, ncolors=len(shap_algo_lst))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=axes, orientation='vertical', pad=0.02, fraction=0.05)
    colorbar.set_ticks(np.arange(len(shap_algo_lst)))
    colorbar.set_ticklabels(shap_algo_lst)

    plt.show() # Chỉ hiển thị, không lưu file