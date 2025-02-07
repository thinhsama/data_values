import torch  # Thư viện xử lý tensor cho máy học
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Subset  # Dùng để tạo batch dữ liệu
import numpy as np  # Thư viện xử lý mảng số
from sklearn.utils import check_random_state  # Kiểm tra và thiết lập random state
import tqdm  # Thư viện hiển thị tiến trình
from tqdm import trange  # Thư viện hiển thị tiến trình
import itertools
from functools import partial
from typing import Callable, Literal, Optional, Union
import geomloss
import ot
from scipy.special import comb
# f1_score
from sklearn.metrics import f1_score
from machine_learning_model import ClassifierMLP, LogisticRegression
def tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1
  if dis_metric == 'cosine':
    if x_test.ndim == 1:
        x_test = x_test.reshape(1, -1)
    dot_product = np.dot(x_train_few, x_test.T)  # Chú ý phải chuyển vị x_val_batch

    # Tính norm của x_train_batch và x_val_batch
    norm_train = np.linalg.norm(x_train_few, axis=1, keepdims=True)
    norm_val = np.linalg.norm(x_test, axis=1, keepdims=True)

    # Tính cosine similarity
    cosine_similarity = dot_product / (norm_train * norm_val.T + 1e-8)

    # Đảo ngược cosine similarity để thành cosine distance (1 - similarity)
    distance = 1 - cosine_similarity
    #distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
  Itau = (distance < tau).nonzero()[0]

  Ct = len(Itau)
  Ca = ( y_train_few[Itau] == y_test ).sum().item()

  reusable_sum = 0
  stable_ratio = 1
  for j in range(N):
    stable_ratio *= (N-j-Ct) / (N-j)
    reusable_sum += (1/(j+1)) * (1 - stable_ratio)
    # reusable_sum += (1/(j+1)) * (1 - comb(N-1-j, Ct) / comb(N, Ct))

  for i in Itau:
    sv[i] = ( int(y_test==y_train_few[i]) - 1/C ) / Ct
    if Ct >= 2:
      ca = Ca - int(y_test==y_train_few[i])
      sv[i] += ( int(y_test==y_train_few[i])/Ct - ca/(Ct*(Ct-1)) ) * ( reusable_sum - 1 )

  return sv

def tnn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, dis_metric='cosine'):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)
  x_train_view, x_valid_view = x_train_few.view(N, -1), x_val_few.view(n_test, -1)
  print('tau in tnn shapley', tau)
  for i in tqdm.tqdm(range(n_test)):
    x_test, y_test = x_valid_view[i], y_val_few[i]
    sv += tnn_shapley_single(x_train_view, y_train_few, x_test, y_test, tau, K0, dis_metric=dis_metric)

  return sv
class TKNN_Shapley():
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor, k_neighbors: int=10, T:int = 0, default: bool = 1, batch_size: int = 32, embedding_model=None, random_state: int=42):
        self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)
        self.T = T  # T = N-2K
        self.default = default

    def train_data_values(self):
        # Preprocess data if embedding model exists
        if self.embedding_model is not None:
            self.x_train, self.x_valid = self.embedding_model(self.x_train, self.x_valid)
        # Compute Shapley values using the tnn_shapley algorithm
        self.data_values = tnn_shapley(
            x_train_few=self.x_train,
            y_train_few=self.y_train,
            x_val_few=self.x_valid,
            y_val_few=self.y_valid,
            tau=self.T,  # or another threshold if needed
            K0=self.k_neighbors,
            dis_metric='cosine'
        )

        return self

    def evaluate_data_values(self) -> np.ndarray:
        return self.data_values
class CKNN_Shapley():
  def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor, k_neighbors: int=10, T:int = 0, default: bool = 1, batch_size: int = 32, embedding_model = None, random_state: int=42):
    self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
    self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
    self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
    self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid
    self.k_neighbors = k_neighbors
    self.batch_size = batch_size
    self.embedding_model = embedding_model
    self.random_state = check_random_state(random_state)
    self.T = T # T = N-2K
    self.default = default
  def match(self, y: torch.Tensor) -> torch.Tensor:
    return (y == self.y_valid).float()
  def train_data_values(self):
    n = len(self.x_train)
    m = len(self.x_valid)
    #self.T = n-self.k_neighbors*2
    #self.T = 1
    if self.default:
        self.T = n//2
    if self.embedding_model is not None:
      self.x_train, self.x_valid = self.embedding_model(self.x_train, self.x_valid)
    #x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
    x_train_view, x_valid_view = self.x_train.view(n, -1), self.x_valid.view(m, -1)
    #print(x_train_view.shape, x_valid_view.shape)
    dist_list = []
    for x_train_batch in DataLoader(x_train_view, batch_size=self.batch_size, shuffle=False):
      dist_row = []
      for x_val_batch in DataLoader(x_valid_view, batch_size=self.batch_size, shuffle=False):
        dist_batch = torch.cdist(x_train_batch, x_val_batch)
        dist_row.append(dist_batch)
      dist_list.append(torch.cat(dist_row, dim=1))
    dist = torch.cat(dist_list, dim=0)
    #print(dist.shape)
    sort_indices = torch.argsort(dist, dim=0, stable=True)
    y_train_sort = self.y_train[sort_indices]
    score = torch.zeros_like(dist)
    for i in tqdm.tqdm(range(n-1, n-self.T-1, -1)):
      score[sort_indices[i], range(m)] = 0
    score[sort_indices[n-self.T-1], range(m)] = self.match(y_train_sort[n-self.T-1])/(n-self.T)
    for i in tqdm.tqdm(range(n-self.T-1, -1, -1)):
      score[sort_indices[i], range(m)] = score[sort_indices[i+1], range(m)] + (self.match(y_train_sort[i]) - self.match(y_train_sort[i+1]))/max(self.k_neighbors, i+1)
    self.data_values = score.mean(axis=1).detach().numpy()
    return self
  def evaluate_data_values(self)->np.ndarray:
    return self.data_values

class KNN_Shapley():
  def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor, k_neighbors: int=10, batch_size: int = 32, metric = 'cosine', embedding_model = None, random_state: int=42):
     # Đảm bảo dữ liệu là tensor PyTorch
    self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
    self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
    self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
    self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid
    self.k_neighbors = k_neighbors
    self.batch_size = batch_size
    self.embedding_model = embedding_model
    self.random_state = check_random_state(random_state)
    self.metric = metric
  def match(self, y: torch.Tensor) -> torch.Tensor:
    return (y == self.y_valid).float()
  def train_data_values(self):
    n = len(self.x_train)
    m = len(self.x_valid)
    #x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
    # 1
    x_train_view, x_valid_view = self.x_train.view(n, -1), self.x_valid.view(m, -1)
    #print(x_train_view.shape, x_valid_view.shape)
    dist_list = []
    for x_train_batch in DataLoader(x_train_view, batch_size=self.batch_size, shuffle=False):
      dist_row = []
      for x_val_batch in DataLoader(x_valid_view, batch_size=self.batch_size, shuffle=False):
        if self.metric == 'cosine':
            dot_product = torch.mm(x_train_batch, x_val_batch.T)  # Chú ý phải chuyển vị x_val_batch

            # Tính norm của x_train_batch và x_val_batch
            norm_train = torch.norm(x_train_batch, dim=1, keepdim=True)
            norm_val = torch.norm(x_val_batch, dim=1, keepdim=True)

            # Tính cosine similarity
            cosine_similarity = dot_product / (norm_train * norm_val.T + 1e-8)

            # Đảo ngược cosine similarity để thành cosine distance (1 - similarity)
            dist_batch = 1 - cosine_similarity
        else:
            dist_batch = torch.cdist(x_train_batch, x_val_batch)
        dist_row.append(dist_batch)
      dist_list.append(torch.cat(dist_row, dim=1))
    dist = torch.cat(dist_list, dim=0)
    #print(dist.shape)
    sort_indices = torch.argsort(dist, dim=0, stable=True)
    #print(sort_indices.shape)
    #print(sort_indices)
    y_train_sort = self.y_train[sort_indices]
    score = torch.zeros_like(dist)
    #print("aaaa", y_train_sort.shape, y_train_sort)
    #print(y_train_sort[n-1])
    score[sort_indices[n-1], range(m)] = self.match(y_train_sort[n-1])/n
    for i in tqdm.tqdm(range(n-2, -1, -1)):
      score[sort_indices[i], range(m)] = score[sort_indices[i+1], range(m)] + (self.match(y_train_sort[i]) - self.match(y_train_sort[i+1]))/max(self.k_neighbors, i+1)
    self.data_values = score.mean(axis=1).detach().numpy()
    return self
  def evaluate_data_values(self)->np.ndarray:
    return self.data_values
class KNNRegression:
    """
    KNN Regression for Data Valuation.
    This class computes data valuation scores for regression problems using K-Nearest Neighbors.

    Parameters
    ----------
    k_neighbors : int
        Number of neighbors to consider.
    batch_size : int
        Batch size for processing.
    device : torch.device
        Device to perform computations (CPU or GPU).
    """
    def __init__(
        self,
        x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor, k_neighbors: int=10, batch_size: int = 32, embedding_model = None, random_state: int=42,
    ):
        # Đảm bảo dữ liệu là tensor PyTorch
        self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)
        

    def compute_distances(self, x_train: torch.Tensor, x_valid: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between training and validation points.

        Parameters
        ----------
        x_train : torch.Tensor
            Training data of shape (n_train, n_features).
        x_valid : torch.Tensor
            Validation data of shape (n_valid, n_features).

        Returns
        -------
        distances : torch.Tensor
            Pairwise distances of shape (n_train, n_valid).
        """
        dist_list = []
        for x_train_batch in DataLoader(x_train, batch_size=self.batch_size):
            dist_row = []
            for x_valid_batch in DataLoader(x_valid, batch_size=self.batch_size):
                dist_row.append(torch.cdist(x_train_batch, x_valid_batch))
            dist_list.append(torch.cat(dist_row, dim=1))
        return torch.cat(dist_list, dim=0)

    def train_data_values(self):
        """
        Compute Shapley values for training data.

        Returns
        -------
        shapley_values : np.ndarray
            Shapley values for each training data point.
        """
        n_train = len(self.x_train)
        n_valid = len(self.x_valid)

        x_train, y_train = self.x_train.view(n_train, -1), self.y_train.view(n_train)
        x_valid, y_valid = self.x_valid.view(n_valid, -1), self.y_valid.view(n_valid)

        # Tính khoảng cách và sắp xếp
        distances = self.compute_distances(x_train, x_valid)
        sort_indices = torch.argsort(distances, dim=0, stable=True)
        y_train_sorted = y_train[sort_indices]

        # Khởi tạo điểm Shapley
        score = torch.zeros_like(distances, dtype=torch.float32)

        # Gán giá trị ban đầu
        score[sort_indices[n_train - 1], torch.arange(n_valid)] = (
            torch.abs(y_train_sorted[n_train - 1, torch.arange(n_valid)] - y_valid) / n_train
        )

        # Tính toán điểm Shapley
        for i in tqdm.tqdm(range(n_train - 2, -1, -1), desc="Computing Shapley Values"):
            score[sort_indices[i], torch.arange(n_valid)] = (
                score[sort_indices[i + 1], torch.arange(n_valid)]
                + (min(self.k_neighbors, i + 1) / (self.k_neighbors * (i + 1)))
                * (
                    torch.abs(y_train_sorted[i, torch.arange(n_valid)] - y_valid)
                    - torch.abs(y_train_sorted[i + 1, torch.arange(n_valid)] - y_valid)
                )
            )

        # Tính giá trị Shapley trung bình
        self.data_values = score.mean(dim=1).cpu().numpy()
        #print('Shapley Values:', shapley_values)
        return self
    def evaluate_data_values(self)->np.ndarray:
        return - self.data_values

# Định nghĩa cost routines cho GeomLoss
cost_routines = {
    1: geomloss.utils.distances,
    2: lambda x, y: geomloss.utils.squared_distances(x, y) / 2,
}
class DatasetDistance_geoloss:
    """
    Lớp tính toán khoảng cách dữ liệu sử dụng Optimal Transport, với hỗ trợ GeomLoss.
    """
    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        feature_cost: Union[Literal["euclidean"], Callable[..., torch.Tensor]] = "euclidean",
        p: int = 2,
        entreg: float = 0.1,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        inner_ot_loss: str = "sinkhorn",
        inner_ot_debiased: bool = False,
        inner_ot_p: int = 2,
        inner_ot_entreg: float = 0.1,
        device: torch.device = torch.device("cpu"),
        ot_method: str = 'balance_ot_sinkhorn',
        label_distances: torch.Tensor = None,
    ):
        self.feature_cost = feature_cost
        self.inner_ot_loss = inner_ot_loss
        self.p = p
        self.entreg = entreg
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.inner_ot_p = inner_ot_p
        self.inner_ot_entreg = inner_ot_entreg
        self.inner_ot_debiased = inner_ot_debiased
        self.device = device
        self.label_distances = label_distances

        # Load datasets
        self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid
        self.num_train, self.num_valid = len(y_train), len(y_valid)

    def _get_label_distances(self) -> torch.Tensor:
        """Precompute label-to-label distances."""
        if self.label_distances is not None:
            return self.label_distances

        pwdist = partial(
            pwdist_exact,
            symmetric=False,
            p=self.inner_ot_p,
            loss=self.inner_ot_loss,
            debias=self.inner_ot_debiased,
            entreg=self.inner_ot_entreg,
            cost_function=self.feature_cost,
            device=self.device,
        )

        DYY1 = pwdist(self.x_train, self.y_train)
        DYY2 = pwdist(self.x_valid, self.y_valid)
        DYY12 = pwdist(self.x_train, self.y_train, self.x_valid, self.y_valid)

        #D = torch.cat([torch.cat([DYY1, DYY12], 1), torch.cat([DYY12.t(), DYY2], 1)])
        # thay ham D
        D = torch.cat([torch.cat([DYY12, DYY1], 1), torch.cat([DYY12.t(), DYY2], 1)])
        #print(D.shape) 2*class
        self.label_distances = D
        return self.label_distances

    def dual_sol(self) -> tuple[float, torch.Tensor]:
        """Compute dataset distance using optimal transport."""
        wasserstein = self._get_label_distances().to(self.device)

        cost_geomloss = partial(
            batch_augmented_cost,
            W=wasserstein,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            feature_cost=self.feature_cost,
        )
        loss = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=self.p,
            cost=cost_geomloss,
            debias=True,
            blur=self.entreg ** (1 / self.p),
            #reach = 0.1,  # Điều chỉnh "mass variation" cho UOT
            backend="tensorized",
        )
        Z1 = torch.cat((self.x_train, self.y_train.float().unsqueeze(dim=1)), -1)
        Z2 = torch.cat((self.x_valid, self.y_valid.float().unsqueeze(dim=1)), -1)
        N, M = len(self.x_train), len(self.x_valid)
        a = torch.ones(N, device=self.device) / N
        b = torch.ones(M, device=self.device) / M
        with torch.no_grad():
            loss.debias = False
            loss.potentials = True
            F_i, G_j = loss(a, Z1.to(self.device), b, Z2.to(self.device))
            pi = [F_i, G_j]

        return pi
    def compute_distance(self, pi) -> np.ndarray:
        # return data values for each traning data point
        # get the clibrated gradient of dual solution = data values
        f1k = pi.squeeze()
        num_points = len(f1k) -1
        train_gradients = f1k
        train_gradients = -1*train_gradients
        return train_gradients.numpy(force = True)
def pwdist_exact(
    X1: torch.Tensor,
    Y1: torch.Tensor,
    X2: Optional[torch.Tensor] = None,
    Y2: Optional[torch.Tensor] = None,
    symmetric: bool = False,
    loss: str = "sinkhorn",
    cost_function: Union[Literal["euclidean"], Callable[..., torch.Tensor]] = "euclidean",
    p: int = 2,
    #reach: float = 0.1,  # Thêm yếu tố Unbalanced OT
    debias: bool = True,
    entreg: float = 1e-1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if X2 is None:
        symmetric = True
        X2, Y2 = X1, Y1

    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    if symmetric:
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        pairs = list(itertools.product(range(n1), range(n2)))

    if cost_function == "euclidean":
        cost_function = cost_routines[p]

    distance = geomloss.SamplesLoss(
        loss=loss,
        p=p,
        cost=cost_function,
        debias=debias,
        blur=entreg ** (1 / p),
        #reach = 0.1,
    )
    D = torch.zeros((n1, n2), device=device, dtype=X1.dtype)
    for i, j in tqdm.tqdm(pairs, leave=False, desc="Computing label-to-label distance"):
        m1 = X1[Y1 == c1[i]].to(device)
        m2 = X2[Y2 == c2[j]].to(device)
           
        a = torch.ones(len(m1), device=device) / len(m1)
        b = torch.ones(len(m2), device=device) / len(m2)
        D[i, j] = distance(a, m1, b, m2).item()
        if c1[i] == c2[j]:
            D[i, j] = D[i, j] * 1
        else:
            D[i, j] = D[i, j] * 1
        if symmetric:
            D[j, i] = D[i, j]

    return D
def batch_augmented_cost(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    feature_cost: Optional[str] = None,
    p: int = 2,
    lam_x: float = 1.0,
    lam_y: float = 1.0,
) -> torch.Tensor:
    Y1 = Z1[:, :, -1].long()
    Y2 = Z2[:, :, -1].long()
    if feature_cost is None or feature_cost == "euclidean":
        C1 = cost_routines[p](Z1[:, :, :-1], Z2[:, :, :-1])
    else:
        C1 = feature_cost(Z1[:, :, :-1], Z2[:, :, :-1])

    if W is not None:
        M = W.shape[1] * Y1[:, :, None] + Y2[:, None, :]
        C2 = W.flatten()[M.flatten(start_dim=1)].reshape(-1, Y1.shape[1], Y2.shape[1])
    else:
        raise ValueError("Must provide label distances or other precomputed metrics")
    return lam_x * C1 + lam_y * (C2 / p)


cost_routines_ot = {
    1: lambda x, y: ot.dist(x.cpu().numpy(), y.cpu().numpy()),  # Sử dụng hàm ot.dist để tính khoảng cách
    2: lambda x, y: ot.dist(x.cpu().numpy(), y.cpu().numpy()) / 2,
}

class DatasetDistance_OT:
    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        feature_cost: Literal["euclidean"] | Callable = "euclidean",
        p: int = 2,
        entreg: float = 0.1,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        inner_ot_loss: str = "sinkhorn",
        inner_ot_debiased: bool = False,
        inner_ot_p: int = 2,
        inner_ot_entreg: float = 0.1,
        device: torch.device = torch.device("cpu"),
        ot_method: str = 'balance_ot_sinkhorn',
        label_distances: torch.Tensor = None,
    ):
        self.label_distances = label_distances
        self.feature_cost = feature_cost
        self.inner_ot_loss = inner_ot_loss
        self.inner_ot_debiased = inner_ot_debiased
        self.inner_ot_p = inner_ot_p
        self.inner_ot_entreg = inner_ot_entreg
        self.device = device
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.p = p
        self.ot_method = ot_method
        self.entreg = entreg
        [*self.covar_dim] = x_train[0].shape
        [*self.label_dim] = (1,) if y_valid.ndim == 1 else y_valid.shape[1:]
        self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid

    def _get_label_distances(self) -> torch.Tensor:
        if self.label_distances is not None:
            return self.label_distances
        pwdist = self._pwdist_exact
        DX = pwdist(self.x_train, self.y_train)
        DY = pwdist(self.x_valid, self.y_valid)
        DXY = pwdist(self.x_train, self.y_train, self.x_valid, self.y_valid)
        D = torch.cat([torch.cat([DX, DXY], 1), torch.cat([DXY.t(), DY], 1)])
        self.label_distances = D
        return D

    def _pwdist_exact(
        self,
        X1: torch.Tensor,
        Y1: torch.Tensor,
        X2: torch.Tensor = None,
        Y2: torch.Tensor = None,
        symetric: bool = False,
        loss: str = 'sinkhorn',
        p: int = 2,
        debias: bool = True,
        entreg: float = 0.1,
        device: torch.device = torch.device('cpu'),
        cost_function=None,
        ot_method: str = 'balance_ot_sinkhorn',
    ) -> torch.Tensor:
        if X2 is None:
            symetric = True
            X2 = X1
            Y2 = Y1
        if isinstance(Y1, np.ndarray):
          Y1 = torch.tensor(Y1, dtype=torch.long, device=device)
        if Y2 is not None and isinstance(Y2, np.ndarray):
          Y2 = torch.tensor(Y2, dtype=torch.long, device=device)
        c1 = torch.unique(Y1)
        c2 = torch.unique(Y2)
        n1, n2 = len(c1), len(c2)
        if symetric:
            pairs = list(itertools.combinations(range(n1), 2))
        else:
            pairs = list(itertools.product(range(n1), range(n2)))
        if cost_function is None:
            cost_function = cost_routines_ot[p]
        D = torch.zeros((n1, n2), device=device)
        for i, j in tqdm.tqdm(pairs, leave=False, desc="computing label-to-label"):
            m1 = X1[Y1 == c1[i]].to(device)
            m2 = X2[Y2 == c2[j]].to(device)

            # Tính khoảng cách Sinkhorn giữa m1 và m2
            cost_matrix = cost_function(m1, m2)  # Khoảng cách giữa các mẫu
            #print('cost_matrix:', cost_matrix)
            
            cost_matrix = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())
            # mean = cost_matrix.mean()
            # std = cost_matrix.std()
            # cost_matrix = (cost_matrix - mean) / (std + 1e-6)
            a, b = ot.unif(len(m1)), ot.unif(len(m2))  # phân phối đều
            #print(f"unif label{i} , unif label:{j}, a{a}, b{b}")
            if self.ot_method == 'balance_ot_sinkhorn':
              D[i, j] = torch.tensor(ot.sinkhorn2(a, b, cost_matrix, entreg,  numItermax=1000), device=device)  # Sinkhorn với entropy regularization
            if self.ot_method == 'unbalance_ot_sinkhorn':
              #D[i, j] = torch.tensor(ot.unbalanced.sinkhorn_unbalanced(a, b, cost_matrix, entreg, reg_m=0.05), device=device)  # Sinkhorn với entropy regularization
              D[i, j] = torch.tensor(ot.sinkhorn2(a, b, cost_matrix, entreg), device=device)  # Sinkhorn với entropy regularization
            if self.ot_method == 'partial':
              D[i, j] = torch.tensor(ot.partial.partial_wasserstein(a, b, cost_matrix, entreg, m=0.7), device=device)
            if symetric:
                D[j, i] = D[i, j]
        return D

    def dual_sol(self) -> tuple[float, torch.Tensor]:
        wassetein = self._get_label_distances().to(self.device)
        def zscore_to_unit_interval(tensor):
          # Tính Z-score
          mean = tensor.mean()
          std = tensor.std()
          
          # Nếu std gần bằng 0, thêm giá trị nhỏ để tránh chia 0
          if std < 1e-6:
              std += 1e-6
          
          zscore = (tensor - mean) / std  # Chuẩn hóa Z-score
          
          # Kiểm tra giá trị min và max trong Z-score
          z_min = zscore.min()
          z_max = zscore.max()
          
          # Nếu min và max gần bằng nhau, thêm epsilon để tránh chia 0
          if z_max - z_min < 1e-6:
              z_max += 1e-6
          
          # Đưa Z-score về [0, 1]
          scaled = (zscore - z_min) / (z_max - z_min)
          scaled = torch.clamp(scaled, 0, 1)  # Đảm bảo giá trị nằm trong khoảng [0, 1]
          
          return scaled

    
          return scaled
        def cost_ot(Z1, Z2, W, lam_x, lam_y, feature_cost, p):
            #if Z1.ndim == 2:
            #    Z1 = Z1.unsqueeze(dim=0)
            #if Z2.ndim == 2:
            #    Z2 = Z2.unsqueeze(dim=0)
            #print(Z1.shape)
            #print(Z2.shape)
            Y1 = Z1[:, -1].long()
            Y2 = Z2[:, -1].long()
            if feature_cost == 'euclidean':
                C1 = cost_routines_ot[p](Z1[:, :-1], Z2[:, :-1])
            else:
                C1 = feature_cost(Z1[:, :-1], Z2[:, :-1])
            if W is not None:
                M = W.shape[1] * Y1[:, None] + Y2[None, :]
                C2 = W.flatten()[M.flatten()].reshape(Y1.shape[0], Y2.shape[0])
            if isinstance(C1, np.ndarray):
              C1 = torch.tensor(C1)
            C1 = zscore_to_unit_interval(C1)
            C2 = zscore_to_unit_interval(C2)
            #print('C1 la:', C1)
            #print('C2 la:', C2)
    

            # Tổng hợp chi phí
            D = C1 * lam_x + (C2 / p) * lam_y
            return D

        Z1 = torch.cat([self.x_train, self.y_train.float().unsqueeze(dim=1)], -1)
        Z2 = torch.cat([self.x_valid, self.y_valid.float().unsqueeze(dim=1)], -1)
        # Tính toán các potential sử dụng Sinkhorn
        cost_matrix = cost_ot(Z1, Z2, wassetein, self.lam_x, self.lam_y, self.feature_cost, self.p)
        a, b = ot.unif(len(Z1)), ot.unif(len(Z2))
        a = torch.tensor(a, device=self.device)
        b = torch.tensor(b, device=self.device)
        a = a.float()
        b = b.float()
        cost_matrix = cost_matrix.float()
        #print('cost_matrix:', cost_matrix)
        cost_matrix = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())
        if self.ot_method == 'balance_ot_sinkhorn':
          pi, log = ot.sinkhorn(a, b, cost_matrix, self.entreg, verbose=False, log=True)
        if self.ot_method == 'unbalance_ot_sinkhorn':
          pi, log = ot.unbalanced.sinkhorn_unbalanced(a, b, cost_matrix, self.entreg, reg_m=0.05, verbose=False, log=True)
        if self.ot_method == 'partial':
          pi, log = ot.partial.partial_wasserstein(a, b, cost_matrix, self.entreg, m=0.8, verbose=False, log=True)
        if self.ot_method =='balance_ot_sinkhorn':
          u = log['u']
          v = log['v']
        else:
          u = log['logu']
          v = log['logv']
        return u, v
    def compute_distance(self, pi) -> np.ndarray:
        # return data values for each traning data point
        # get the clibrated gradient of dual solution = data values
        f1k = pi.squeeze()
        num_points = len(f1k) -1
        train_gradients = f1k
        train_gradients = -1*train_gradients
        return train_gradients.numpy(force = True)


class Hier_DatasetDistance_OT:
    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        feature_cost: Literal["euclidean"] | Callable = "euclidean",
        p: int = 2,
        entreg: float = 0.1,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        inner_ot_loss: str = "sinkhorn",
        inner_ot_debiased: bool = False,
        inner_ot_p: int = 2,
        inner_ot_entreg: float = 0.1,
        device: torch.device = torch.device("cpu"),
        ot_method: str = 'balance_ot_sinkhorn',
    ):
        self.feature_cost = feature_cost
        self.inner_ot_loss = inner_ot_loss
        self.inner_ot_debiased = inner_ot_debiased
        self.inner_ot_p = inner_ot_p
        self.inner_ot_entreg = inner_ot_entreg
        self.device = device
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.p = p
        self.ot_method = ot_method
        self.entreg = entreg
        [*self.covar_dim] = x_train[0].shape
        [*self.label_dim] = (1,) if y_valid.ndim == 1 else y_valid.shape[1:]
        self.label_distances = None
        self.real_label = None
        self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid

    def _get_label_distances(self) -> torch.Tensor:
        if self.label_distances is not None:
            return self.label_distances
        pwdist = self._pwdist_exact
        DX = pwdist(self.x_train, self.y_train)
        DY = pwdist(self.x_valid, self.y_valid)
        DXY = pwdist(self.x_train, self.y_train, self.x_valid, self.y_valid)
        D = torch.cat([torch.cat([DX, DXY], 1), torch.cat([DXY.t(), DY], 1)])
        self.label_distances = D
        self.real_label = DXY
        return D

    def _pwdist_exact(
        self,
        X1: torch.Tensor,
        Y1: torch.Tensor,
        X2: torch.Tensor = None,
        Y2: torch.Tensor = None,
        symetric: bool = False,
        loss: str = 'sinkhorn',
        p: int = 2,
        debias: bool = True,
        entreg: float = 0.1,
        device: torch.device = torch.device('cpu'),
        cost_function=None,
        ot_method: str = 'balance_ot_sinkhorn',
    ) -> torch.Tensor:
        if X2 is None:
            symetric = True
            X2 = X1
            Y2 = Y1
        if isinstance(Y1, np.ndarray):
          Y1 = torch.tensor(Y1, dtype=torch.long, device=device)
        if Y2 is not None and isinstance(Y2, np.ndarray):
          Y2 = torch.tensor(Y2, dtype=torch.long, device=device)
        c1 = torch.unique(Y1)
        c2 = torch.unique(Y2)
        n1, n2 = len(c1), len(c2)
        if symetric:
            pairs = list(itertools.combinations(range(n1), 2))
        else:
            pairs = list(itertools.product(range(n1), range(n2)))
        if cost_function is None:
            cost_function = cost_routines_ot[p]
        D = torch.zeros((n1, n2), device=device)
        for i, j in tqdm.tqdm(pairs, leave=False, desc="computing label-to-label"):
            m1 = X1[Y1 == c1[i]].to(device)
            m2 = X2[Y2 == c2[j]].to(device)

            # Tính khoảng cách Sinkhorn giữa m1 và m2
            cost_matrix = cost_function(m1, m2)  # Khoảng cách giữa các mẫu
            #print('cost_matrix:', cost_matrix)
            
            cost_matrix = (cost_matrix) / (cost_matrix.max())
            a, b = ot.unif(len(m1)), ot.unif(len(m2))  # phân phối đều
            #print(f"unif label{i} , unif label:{j}, a{a}, b{b}")
            if self.ot_method == 'balance_ot_sinkhorn':
              D[i, j] = torch.tensor(ot.sinkhorn2(a, b, cost_matrix, entreg,  numItermax=1000), device=device)  # Sinkhorn với entropy regularization
              #print('thang thu i,j:', i, j, D[i, j])
            #if ot_method == 'balance_ot_sinkhorn':
            if self.ot_method == 'unbalance_ot_sinkhorn':
              #D[i, j] = torch.tensor(ot.unbalanced.sinkhorn_unbalanced(a, b, cost_matrix, entreg, reg_m=0.05), device=device)  # Sinkhorn với entropy regularization
              D[i, j] = torch.tensor(ot.sinkhorn2(a, b, cost_matrix, entreg), device=device)  # Sinkhorn với entropy regularization
            if self.ot_method == 'partial':
              D[i, j] = torch.tensor(ot.partial.partial_wasserstein(a, b, cost_matrix, entreg, m=0.7), device=device)
            if symetric:
                D[j, i] = D[i, j]
        return D

    def dual_sol(self) -> tuple[float, torch.Tensor]:
        wassetein = self._get_label_distances().to(self.device)
        def zscore_to_unit_interval(tensor):
          # Tính Z-score
          mean = tensor.mean()
          std = tensor.std()
          
          # Nếu std gần bằng 0, thêm giá trị nhỏ để tránh chia 0
          if std < 1e-6:
              std += 1e-6
          
          zscore = (tensor - mean) / std  # Chuẩn hóa Z-score
          
          # Kiểm tra giá trị min và max trong Z-score
          z_min = zscore.min()
          z_max = zscore.max()
          
          # Nếu min và max gần bằng nhau, thêm epsilon để tránh chia 0
          if z_max - z_min < 1e-6:
              z_max += 1e-6
          
          # Đưa Z-score về [0, 1]
          scaled = (zscore - z_min) / (z_max - z_min)
          scaled = torch.clamp(scaled, 0, 1)  # Đảm bảo giá trị nằm trong khoảng [0, 1]
          
          return scaled

    
          return scaled
        def cost_ot(Z1, Z2, W, lam_x, lam_y, feature_cost, p):
            Y1 = Z1[:, -1].long()
            Y2 = Z2[:, -1].long()
            Z1_ot = Z1.unsqueeze(0) # (1, n1, d+1)
            Z2_ot = Z2.unsqueeze(0) # (1, n1, d+1)
            num_classes_Y1 = len(np.unique(Y1.cpu().numpy()))
            num_classes_Y2 = len(np.unique(Y2.cpu().numpy()))
            D = batch_augmented_cost(Z1_ot, Z2_ot, W=W, lam_x = lam_x, lam_y = lam_y)
            D = D.squeeze(0)
            return D

        Z1 = torch.cat([self.x_train, self.y_train.float().unsqueeze(dim=1)], -1)
        Z2 = torch.cat([self.x_valid, self.y_valid.float().unsqueeze(dim=1)], -1)
        # tinh gamma
        # Tính toán các potential sử dụng Sinkhorn
        cost_matrix = cost_ot(Z1, Z2, wassetein, self.lam_x, self.lam_y, self.feature_cost, self.p)
        a, b = ot.unif(len(Z1)), ot.unif(len(Z2))
        a = torch.tensor(a, device=self.device)
        b = torch.tensor(b, device=self.device)
        a = a.float()
        b = b.float()
        cost_matrix = cost_matrix.float()
        #print(cost_matrix.shape, a.shape, b.shape)
        #print('cost_matrix:', cost_matrix)
        cost_matrix = (cost_matrix) / (cost_matrix.max())
        #pi = torch.tensor(ot.sinkhorn2(a, b, cost_matrix, self.entreg, numItermax=1000), device=self.device) 
        if self.ot_method == 'balance_ot_sinkhorn':
          gamma, log = ot.sinkhorn(a, b, cost_matrix, self.entreg, verbose=False, log=True)
        if self.ot_method == 'unbalance_ot_sinkhorn':
          gamma, log = ot.unbalanced.sinkhorn_unbalanced(a, b, cost_matrix, self.entreg, reg_m=0.05, verbose=False, log=True)
        if self.ot_method == 'partial':
          gamma, log = ot.partial.partial_wasserstein(a, b, cost_matrix, self.entreg, m=0.8, verbose=False, log=True)
        if self.ot_method =='balance_ot_sinkhorn':
          u = log['u']
          v = log['v']
        else:
          u = log['logu']
          v = log['logv']
        return cost_matrix, gamma, [u, v] 
    def compute_distance(self, pi) -> np.ndarray:
      # return data values for each traning data point
      # get the clibrated gradient of dual solution = data values
      f1k = pi.squeeze()
      num_points = len(f1k) -1
      train_gradients = f1k*(1+1/(num_points)) - f1k.sum()/num_points
      train_gradients = -1*train_gradients
      return train_gradients.numpy(force = True)
    


class HierDatasetDistance_geoloss:
    """
    Lớp tính toán khoảng cách dữ liệu sử dụng Optimal Transport, với hỗ trợ GeomLoss.
    """
    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        feature_cost: Union[Literal["euclidean"], Callable[..., torch.Tensor]] = "euclidean",
        p: int = 2,
        entreg: float = 0.1,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        inner_ot_loss: str = "sinkhorn",
        inner_ot_debiased: bool = False,
        inner_ot_p: int = 2,
        inner_ot_entreg: float = 0.1,
        device: torch.device = torch.device("cpu"),
        ot_method: str = 'balance_ot_sinkhorn',
    ):
        self.feature_cost = feature_cost
        self.inner_ot_loss = inner_ot_loss
        self.p = p
        self.entreg = entreg
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.inner_ot_p = inner_ot_p
        self.inner_ot_entreg = inner_ot_entreg
        self.inner_ot_debiased = inner_ot_debiased
        self.device = device
        self.label_distances = None

        # Load datasets
        self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid
        self.num_train, self.num_valid = len(y_train), len(y_valid)

    def _get_label_distances(self) -> torch.Tensor:
        """Precompute label-to-label distances."""
        if self.label_distances is not None:
            return self.label_distances

        pwdist = partial(
            pwdist_exact,
            symmetric=False,
            p=self.inner_ot_p,
            loss=self.inner_ot_loss,
            debias=self.inner_ot_debiased,
            entreg=self.inner_ot_entreg,
            cost_function=self.feature_cost,
            device=self.device,
        )

        DYY1 = pwdist(self.x_train, self.y_train)
        DYY2 = pwdist(self.x_valid, self.y_valid)
        DYY12 = pwdist(self.x_train, self.y_train, self.x_valid, self.y_valid)
        D = torch.cat([torch.cat([DYY12, DYY1], 1), torch.cat([DYY12.t(), DYY2], 1)])
        #print(D.shape) 2*class
        self.label_distances = D
        return self.label_distances

    def dual_sol(self) -> tuple[float, torch.Tensor]:
        """Compute dataset distance using optimal transport."""
        wasserstein = self._get_label_distances().to(self.device)

        cost_geomloss = partial(
            batch_augmented_cost,
            W=wasserstein,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            feature_cost=self.feature_cost,
        )
        loss = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=self.p,
            cost=cost_geomloss,
            debias=True,
            blur=self.entreg ** (1 / self.p),
            #reach = 0.1,  # Điều chỉnh "mass variation" cho UOT
            backend="tensorized",
        )
        Z1 = torch.cat((self.x_train, self.y_train.float().unsqueeze(dim=1)), -1)
        Z2 = torch.cat((self.x_valid, self.y_valid.float().unsqueeze(dim=1)), -1)
        Z1_ot = Z1.unsqueeze(0) # (1, n1, d+1)
        Z2_ot = Z2.unsqueeze(0) # (1, n1, d+1)
        costs = batch_augmented_cost(
                Z1_ot,
                Z2_ot,
                W=wasserstein,
                lam_x=self.lam_x,
                lam_y=self.lam_y,
                feature_cost=self.feature_cost,
            )
        C = costs.squeeze()
        N, M = len(self.x_train), len(self.x_valid)
        a = torch.ones(N, device=self.device) / N
        b = torch.ones(M, device=self.device) / M
        gamma = ot.sinkhorn(a, b, C/C.max(), self.entreg, verbose=False)
        with torch.no_grad():
            loss.debias = False
            loss.potentials = True
            F_i, G_j = loss(a, Z1.to(self.device), b, Z2.to(self.device))
            pi = [F_i, G_j]

        return costs, gamma, pi
    def compute_distance(self, pi) -> np.ndarray:
        # return data values for each traning data point
        # get the clibrated gradient of dual solution = data values
        f1k = pi.squeeze()
        num_points = len(f1k) -1
        train_gradients = f1k
        train_gradients = -1*train_gradients
        return train_gradients.numpy(force = True)






class Hier_SAVA_OT:
    def __init__(self, random_state=0, batch_size=100, device=torch.device("cpu")):
        self.random_state = random_state
        self.device = device
        torch.manual_seed(random_state)
        self.batch_size = batch_size
    def create_dataloaders(self, x_train, y_train, x_valid, y_valid):
        """
        Create DataLoaders for train and validation datasets.
        """
        # Convert inputs to tensors if they are not already
        x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid

        # Create TensorDatasets
        train_dataset = TensorDataset(x_train, y_train)
        valid_dataset = TensorDataset(x_valid, y_valid)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    def get_calibrated_gradients(self, dual_sol, training_size):
        f1k = np.array(dual_sol[0].squeeze().cpu()) # dual_sol: tuple len(2) dual_sol[0] = [tr_size, 1] dual
        train_gradient = [0] * training_size
        train_gradient = (1 + 1 / (training_size - 1)) * f1k - sum(f1k) / (training_size - 1)
        return list(train_gradient)

    def evaluate_data_values(self, x_train, y_train, x_valid, y_valid, lam_x=1.0, lam_y=1.0):
        """
        Train model and evaluate data values based on OT.
        """
        self.create_dataloaders(x_train, y_train, x_valid, y_valid)

        # Tạo dictionary cho dual solutions
        dual_sol_dict = {i: {} for i in range(len(self.train_loader))}
        costs_bar = np.zeros((len(self.train_loader), len(self.valid_loader)))

        for i, (x_train_batch, y_train_batch) in enumerate(tqdm.tqdm(self.train_loader, desc="batch LAVA")):
            for j, (x_valid_batch, y_valid_batch) in enumerate(self.valid_loader):
                # Tính chi phí OT và các dual solutions
                dist_calculator = HierDatasetDistance_geoloss(
                    x_train_batch.to(self.device),
                    y_train_batch.to(self.device),
                    x_valid_batch.to(self.device),
                    y_valid_batch.to(self.device),
                    device=self.device,
                    ot_method="balance_ot_sinkhorn"
                )
                cost, plan, dual_sol = dist_calculator.dual_sol()
                costs_bar[i, j] = 1 / (x_train_batch.shape[0] + x_valid_batch.shape[0]) * torch.sum(plan * cost).item()
                dual_sol_dict[i][j] = dual_sol

        # Tính gamma_bar
        a = np.ones(costs_bar.shape[0])
        b = np.ones(costs_bar.shape[1])
        eps = np.max(costs_bar)
        gamma_bar = ot.sinkhorn(a, b, costs_bar / eps, 1e-02, verbose=False)
        #gamma_bar_test = np.ones_like(gamma_bar)
        # Tính trọng số điểm dữ liệu
        values = []
        for k, (x_train_batch, y_train_batch) in enumerate(tqdm.tqdm(self.train_loader, desc="batch LAVA")):
            for l in range(x_train_batch.shape[0]):
                threshold_gradients = np.zeros(len(self.valid_loader))
                for m in range(len(self.valid_loader)):
                    dual_sol = dual_sol_dict[k][m]
                    calibrated_gradient = self.get_calibrated_gradients(dual_sol, training_size=x_train_batch.shape[0])
                    threshold_gradients[m] = calibrated_gradient[l]

                s_l = np.sum(gamma_bar[k] * threshold_gradients)
                values.append((-1)*s_l)

        return np.array(values)





class SAVA_OT:
    def __init__(self, random_state=0, batch_size=100, device=torch.device("cpu")):
        self.random_state = random_state
        self.device = device
        torch.manual_seed(random_state)
        self.batch_size = batch_size
    def create_dataloaders(self, x_train, y_train, x_valid, y_valid):
        """
        Create DataLoaders for train and validation datasets.
        """
        # Convert inputs to tensors if they are not already
        x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid

        # Create TensorDatasets
        train_dataset = TensorDataset(x_train, y_train)
        valid_dataset = TensorDataset(x_valid, y_valid)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    def evaluate_data_values(self, x_train, y_train, x_valid, y_valid, lam_x=1.0, lam_y=1.0):
        # Train model and evaluate data values based on OT
        """
        Train model and evaluate data values based on OT.
        """
        # Create DataLoaders
        self.create_dataloaders(x_train, y_train, x_valid, y_valid)
        values = []
        for i, (x_train_batch, y_train_batch) in enumerate(tqdm.tqdm(self.train_loader, desc='batch LAVA')):
            # Move data to the specified device
            x_train_batch = x_train_batch.to(self.device)
            y_train_batch = y_train_batch.to(self.device)
            values_tmp = np.zeros(x_train_batch.shape[0])
            for j, (x_valid_batch, y_valid_batch) in enumerate(tqdm.tqdm(self.valid_loader, desc = 'batch LAVA')):
                # Move data to the specified device
                x_valid_batch = x_valid_batch.to(self.device)
                y_valid_batch = y_valid_batch.to(self.device)
                dist_calculator = DatasetDistance_geoloss(x_train_batch, y_train_batch, x_valid_batch, y_valid_batch, device=self.device, ot_method='balance_ot_sinkhorn')
                u = dist_calculator.dual_sol()
                # dist dua tren calibrate gradient
                dist = dist_calculator.compute_distance(u[0])
                values_tmp += dist 
            values.append(values_tmp/len(self.valid_loader))
        values = np.concatenate(values)
        return values




class SAVA_OT_savel2l:
    def __init__(self, random_state=0, batch_size=100, device=torch.device("cpu")):
        self.random_state = random_state
        self.device = device
        torch.manual_seed(random_state)
        self.batch_size = batch_size
    def create_dataloaders(self, x_train, y_train, x_valid, y_valid):
        """
        Create DataLoaders for train and validation datasets.
        """
        # Convert inputs to tensors if they are not already
        x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid

        # Create TensorDatasets
        train_dataset = TensorDataset(x_train, y_train)
        valid_dataset = TensorDataset(x_valid, y_valid)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    def evaluate_data_values(self, x_train, y_train, x_valid, y_valid, lam_x=1.0, lam_y=1.0):
        # Train model and evaluate data values based on OT
        """
        Train model and evaluate data values based on OT.
        """
        # Create DataLoaders
        self.create_dataloaders(x_train, y_train, x_valid, y_valid)
        dist_calculator = DatasetDistance_geoloss(x_train, y_train, x_valid, y_valid, device=self.device, ot_method='balance_ot_sinkhorn')
        label_distances = dist_calculator._get_label_distances()
        values = []
        for i, (x_train_batch, y_train_batch) in enumerate(tqdm.tqdm(self.train_loader, desc='batch LAVA')):
            # Move data to the specified device
            x_train_batch = x_train_batch.to(self.device)
            y_train_batch = y_train_batch.to(self.device)
            values_tmp = np.zeros(x_train_batch.shape[0])
            for j, (x_valid_batch, y_valid_batch) in enumerate(tqdm.tqdm(self.valid_loader, desc = 'batch LAVA')):
                # Move data to the specified device
                x_valid_batch = x_valid_batch.to(self.device)
                y_valid_batch = y_valid_batch.to(self.device)
                dist_calculator = DatasetDistance_geoloss(x_train_batch, y_train_batch, x_valid_batch, y_valid_batch, label_distances=label_distances ,device=self.device, ot_method='balance_ot_sinkhorn')
                u = dist_calculator.dual_sol()
                # dist dua tren calibrate gradient
                dist = dist_calculator.compute_distance(u[0])
                #assert dist[0] == x_train_batch.shape[0]
                values_tmp += dist 
            values.append(values_tmp/len(self.valid_loader))
        values = np.concatenate(values)
        return values

class SAVA_OT_tanh:
    def __init__(self, random_state=0, batch_size=100, device=torch.device("cpu")):
        self.random_state = random_state
        self.device = device
        torch.manual_seed(random_state)
        self.batch_size = batch_size
    def create_dataloaders(self, x_train, y_train, x_valid, y_valid):
        """
        Create DataLoaders for train and validation datasets.
        """
        # Convert inputs to tensors if they are not already
        x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid

        # Create TensorDatasets
        train_dataset = TensorDataset(x_train, y_train)
        valid_dataset = TensorDataset(x_valid, y_valid)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    def evaluate_data_values(self, x_train, y_train, x_valid, y_valid, lam_x=1.0, lam_y=1.0):
        # Train model and evaluate data values based on OT
        """
        Train model and evaluate data values based on OT.
        """
        # Create DataLoaders
        self.create_dataloaders(x_train, y_train, x_valid, y_valid)
        values = []
        for i, (x_train_batch, y_train_batch) in enumerate(tqdm.tqdm(self.train_loader, desc='batch LAVA')):
            # Move data to the specified device
            x_train_batch = x_train_batch.to(self.device)
            y_train_batch = y_train_batch.to(self.device)
            values_tmp = np.zeros(x_train_batch.shape[0])
            for j, (x_valid_batch, y_valid_batch) in enumerate(tqdm.tqdm(self.valid_loader, desc = 'batch LAVA')):
                # Move data to the specified device
                x_valid_batch = x_valid_batch.to(self.device)
                y_valid_batch = y_valid_batch.to(self.device)
                dist_calculator = DatasetDistance_OT(x_train_batch, y_train_batch, x_valid_batch, y_valid_batch, device=self.device, ot_method='balance_ot_sinkhorn')
                u = dist_calculator.dual_sol()
                # dist dua tren calibrate gradient
                dist = dist_calculator.compute_distance(u[0])
                #assert u == x_train_batch.shape[0]
                squashed_calibrated_gradient = (np.tanh(dist) + 1) / 2
                values_tmp += squashed_calibrated_gradient / squashed_calibrated_gradient.sum()
            values.append(values_tmp/len(self.valid_loader))
        values = np.concatenate(values)
        return values

class TMCSampler:
    def __init__(self, mc_epochs: int = 100, min_cardinality: int = 5, random_state: int = 42):
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.random_state = np.random.RandomState(random_state)
        self.marginal_contrib_sum = None
        self.marginal_count = None

    def set_coalition(self, num_points: int):
        self.num_points = num_points
        self.marginal_contrib_sum = np.zeros((num_points,))
        self.marginal_count = np.zeros((num_points,))

    def compute_marginal_contribution(self, utility_func):
        for _ in tqdm.trange(self.mc_epochs, desc="Computing TMC Contributions"):
            perm = self.random_state.permutation(self.num_points)
            marginal_increment = 0
            for i in range(self.num_points):
                idx = perm[i]
                new_utility = utility_func(perm[:i+1])
                self.marginal_contrib_sum[idx] += new_utility - marginal_increment
                self.marginal_count[idx] += 1
                marginal_increment = new_utility
        return self.marginal_contrib_sum / self.marginal_count


class ClassWiseShapley:
    def __init__(self, sampler: Optional[TMCSampler] = None, model=None, random_state: int = 42):
        self.sampler = sampler or TMCSampler(random_state=random_state)
        self.model = model  # Thêm mô hình vào class

    def input_data(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor):
        self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid
        self.train_classes = y_train if y_train.ndim == 1 else torch.argmax(y_train, dim=1)
        self.classes = torch.unique(self.train_classes)
        self.data_values = np.zeros((len(x_train),))
        return self

    def train_data_values(self):
        for label in self.classes:
            in_class_idx = (self.train_classes == label).nonzero(as_tuple=True)[0]
            out_class_idx = (self.train_classes != label).nonzero(as_tuple=True)[0]

            x_in_tensor = torch.stack([self.x_train[i] for i in in_class_idx])
            y_in_tensor = torch.stack([self.y_train[i] for i in in_class_idx])

            x_out_tensor = torch.stack([self.x_train[i] for i in out_class_idx])
            y_out_tensor = torch.stack([self.y_train[i] for i in out_class_idx])

            def utility_func(subset_idx):
              x_comb = torch.cat([x_in_tensor[i].unsqueeze(0) for i in subset_idx] + [x_out_tensor])
              y_comb = torch.cat([y_in_tensor[i].unsqueeze(0) for i in subset_idx] + [y_out_tensor])
              self.model.fit(x_comb, y_comb, epochs=100, lr=0.01)  # Huấn luyện nhanh
              predictions = self.model.predict(self.x_valid)
              return f1_score(self.y_valid.numpy(), predictions.numpy(), average="weighted")

            self.sampler.set_coalition(len(in_class_idx))
            contrib = self.sampler.compute_marginal_contribution(utility_func)
            self.data_values[in_class_idx] += contrib

        return self
    def evaluate_data_values(self):
        return self.data_values
    
class BetaShapley:
    """
    Beta Shapley implementation integrated with TMCSampler.
    """

    def __init__(
        self,
        model,
        sampler: TMCSampler,
        alpha: int = 4,
        beta: int = 1,
    ):
        self.model = model
        self.sampler = sampler
        self.alpha = alpha
        self.beta = beta
        self.data_values = None

    def input_data(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor):
        """
        Input training data and utility function.
        """
        self.x_train = torch.tensor(x_train, dtype=torch.float32) if not isinstance(x_train, torch.Tensor) else x_train
        self.y_train = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train
        self.x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
        self.y_valid = torch.tensor(y_valid, dtype=torch.long) if not isinstance(y_valid, torch.Tensor) else y_valid
        self.num_points = len(x_train)
        self.sampler.set_coalition(self.num_points)
        self.data_values = np.zeros(self.num_points)

    def compute_weights(self):
        """
        Compute Beta Shapley weights for each subset size.
        """
        weights = np.array(
            [
                np.exp(
                    np.log(comb(self.num_points - 1, j))
                    + np.log(self.beta_distribution(j + 1))
                )
                for j in range(self.num_points)
            ]
        )
        return weights / weights.sum()

    def beta_distribution(self, j: int):
        """
        Compute Beta distribution value for a given subset size.
        """
        from scipy.special import beta

        return beta(j + self.alpha, self.num_points - j + self.beta)

    def train_data_values(self):
        """
        Train Beta Shapley values.
        """
        weights = self.compute_weights()

        for epoch in trange(self.sampler.mc_epochs, desc="Computing Beta Shapley Values"):
            perm = self.sampler.random_state.permutation(self.num_points)
            marginal_contrib = np.zeros(self.num_points)
            cumulative_utility = 0

            for i in range(self.num_points):
                idx = perm[i]
                subset_idx = perm[: i + 1]
                def utility_func(subset_idx):
                    self.model.fit(self.x_train[subset_idx], self.y_train[subset_idx], epochs=100, lr=0.01)  # Huấn luyện nhanh
                    predictions = self.model.predict(self.x_valid)
                    return f1_score(self.y_valid.numpy(), predictions.numpy(), average="weighted")
                utility = utility_func(subset_idx)
                marginal_contrib[idx] = utility - cumulative_utility
                cumulative_utility = utility

            self.data_values += weights * marginal_contrib

        self.data_values /= self.sampler.mc_epochs
        return self

    def evaluate_data_values(self):
        """
        Return computed Beta Shapley data values.
        """
        if self.data_values is None:
            raise ValueError("Data values have not been computed. Call `train_data_values` first.")
        return self.data_values