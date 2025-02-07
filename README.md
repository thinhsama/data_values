# Data Valuation and Experimental Framework

## Installation
To install the necessary dependencies, use the following command:
```bash
pip install -r requirements.txt
```
The required libraries include:
- `geomloss==0.2.6`
- `POT==0.9.5`

## Project Structure
This repository contains implementations of various data valuation algorithms, experimental setups, and dataset utilities. Below is an overview of the files and their functionalities:

### Core Algorithms
- **`data_valuate.py`**: Implements data valuation algorithms:
  - `CKNN_Shapley`: Improved KNN-Shapley
  - `KNN_Shapley`: Traditional KNN-Shapley
  - `KNN_regression`: Proposed KNN-Shapley for regression
  - `DatasetDistance_geoloss`: Traditional LAVA
  - `DatasetDistance_OT`: LAVA implemented with OT library
  - `SAVA_OT`: Proposed batch LAVA
  - `SAVA_OT_savel2l`: Batch LAVA with precomputed labels
  - `TMCSampler`: Data Shapley
  - `ClassWiseShapley`: CS Shapley
  - `BetaShapley`: Beta Shapley

### Supporting Modules
- **`base_evaluator.py`**: Bridges algorithms and evaluation
  - `BaseEvaluator`: Common `evaluate_data_values` function
  - `ExperimentRunner`: Runs experiments across multiple algorithms
    - `calculate_label_noise_20`: Selects 20% noisy data for F1-score
    - `calculate_label_noise`: Uses a threshold (<0) for F1-score
    - `calculate_WAD`: Computes WAD score
    - `evaluate`: Generates result tables
    - `plot_results`: Visualizes results
    - `save_results`: Saves results
    - `_get_evaluator_name`: Retrieves evaluator names
    - `run`: Executes data valuation algorithms

### Datasets
- **`datasets.py`**: General dataset utilities
- **`img_dataset.py`**: Image dataset (Torch Vision)
- **`nlp_dataset.py`**: NLP dataset
- **`regression_dataset.py`**: Regression dataset
- **`subset_data.py`**: Creates dataset subsets

### Machine Learning & Data Processing
- **`embedding_img.py`**: Image embedding customization
- **`machine_learning_model.py`**: Machine learning model training algorithms
- **`nosify.py`**: Methods for adding noise to data
- **`other_detect_noise.py`**: Additional noise detection methods

### Experimentation
- **`experiment_method.py`**: Runs experiments comparing F1-score, WAD score
- **`experiment_run.py`**: Compares adding/removing data in experiments
- **`visualize.py`**: Visualization of noisy/error-affected images
- **`parameter.py`**: API definitions

## Experimentation Notebooks
These Jupyter notebooks provide structured experimental setups:
- **`yte.ipynb`**: Medical experiment
- **`unbalance_cifar.ipynb`**: Imbalanced CIFAR dataset & KNN-Shapley parameter tuning
- **`time_series.ipynb`**: Time-series experiment
- **`table_data.ipynb`**: Table data experiment
- **`regression.ipynb`**: Regression experiment
- **`NLP.ipynb`**: NLP experiment
- **`noise_tool.ipynb`**: Comparison with other noise detection tools

### Summary Results
- **`all_algorithm1.ipynb`**: Aggregated results of all algorithms on primary datasets
- **`all_algorithm.ipynb`**: Aggregated results of all algorithms on a primary dataset

## Sample Experimental Workflow
### Preparing dataset
```python
### Load x_y_embedding_data
import pickle
with open('x_y_embedding_cifar_lon.pkl', 'rb') as f:
    #x_embeddings, y_labels, xt_embeddings, yt_labels = pickle.load(f)
    embedding_default_train, finetuned_label_train, embedding_default_valid, finetuned_label_valid, embedding_default_test, finetuned_label_test = pickle.load(f)
x_embeddings = embedding_default_train
y_labels = finetuned_label_train
xt_embeddings = embedding_default_test
yt_labels = finetuned_label_test
print("Training embeddings shape:", x_embeddings.shape)
print("Training labels shape:", y_labels.shape)
print("Validation embeddings shape:", xt_embeddings.shape)
print("Validation labels shape:", yt_labels.shape)

```
### Adding Noise
```python
from machine_learning_model import LogisticRegression
from nosify import mix_label

# Introduce noise into labels
y_copy = y_labels.copy()
yt_copy = yt_labels.copy()
param = mix_label(y_copy, yt_copy, noise_rate=0.2)
y_labels_noisy = param['y_train']
noisy_train_indices = param['noisy_train_indices']
print("Noisy training labels shape:", y_labels_noisy.shape)
```

### Training a Model
```python
# Train logistic regression model
input_dim = x_embeddings.shape[1]
num_classes = len(np.unique(y_labels))
model = LogisticRegression(input_dim, num_classes)
model.fit(x_embeddings, y_labels_noisy, epochs=1000, lr=0.1)

# Predict
y_pred = model.predict(xt_embeddings)
```

### Evaluation
```python
from sklearn.metrics import f1_score
accuracy = f1_score(yt_labels, y_pred, average='weighted')
print("Accuracy:", accuracy)
```
### calling algorithms
```python
from base_evaluator import BaseEvaluator, KNNEvaluator,CKNNEvaluator ,LavaEvaluator_geomloss, LavaEvaluator_OT, ExperimentRunner, LavaEvaluator_batch
knn_evaluator = KNNEvaluator()
cknn_evaluator1 = CKNNEvaluator(T = 20, default=False)
cknn_evaluator2 = CKNNEvaluator(T = 300, default=False)
cknn_evaluator3 = CKNNEvaluator(T = 700, default=False)
lava_evaluator_batch1 = LavaEvaluator_batch(batch = 50)
lava_evaluator_batch2 = LavaEvaluator_batch(batch = 500)
lava_evaluator_OT = LavaEvaluator_OT()
lava_evaluator_geomloss = LavaEvaluator_geomloss()
```
### Running Experiments
```python
experiment = ExperimentRunner(evaluators=[
    knn_evaluator, cknn_evaluator1, cknn_evaluator2,
    cknn_evaluator3, lava_evaluator_batch1, lava_evaluator_batch2,
    lava_evaluator_OT, lava_evaluator_geomloss
])

results = experiment.run(x_embeddings, y_labels_noisy, xt_embeddings, yt_labels)
experiment.evaluate(noisy_train_indices)
```

### Computing F1 Score for Noisy Data
```python
experiment.calculate_label_noise_20(model, noisy_train_indices, 0.3)
```

### API Usage (preparing release)
The API is defined in `parameter.py` and can be called accordingly.

---
This repository provides a structured framework for evaluating data quality, running machine learning experiments, and analyzing dataset reliability using multiple state-of-the-art techniques.





