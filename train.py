import numpy as np
import dill


class LogisticRegression(object):
  def __init__(self):
    self.alpha = None

  def _sigm(self, x: float) -> float:
    return (1 + np.exp(-x)) ** -1

  def _gradient(self, y_true: int, y_pred: float, x: float, betta: np.array) -> np.array:
    x = np.array([x, 1])
    grad = x * (y_pred - y_true) + betta * self.alpha
    return grad

  def fit(self, x_train: np.array, y_train: np.array, lr: float=0.01, betta: np.array=np.array([0, 0]), num_epoch: int=10):
    self.alpha = np.ones(2)
    for epo in range(num_epoch):
      for i,x in enumerate(x_train):
        x0 = np.array([x, 1]) * self.alpha
        y_pred = self._sigm(x0.sum())
        grad = self._gradient(y_train[i], y_pred, x, betta)
        self.alpha -= lr * grad

  def predict(self, x: np.array) -> np.array:
    preds = np.zeros(len(x))
    for i,x0 in enumerate(x):
      x0 = np.array([x0, 1]) * self.alpha
      preds[i] = self._sigm(x0.sum())
    return preds

def precision(y_true: np.array, y_pred: np.array):
  t = np.where(y_pred == y_true)
  f = np.where(y_pred != y_true)
  p = np.where(y_pred == 1)
  tp = len(np.intersect1d(p, t))
  fp = len(np.intersect1d(p, f))
  return tp / (tp + fp)
def recall(y_true: np.array, y_pred: np.array):
  t = np.where(y_pred == y_true)
  f = np.where(y_pred != y_true)
  p = np.where(y_pred == 1)
  n = np.where(y_pred == 0)
  tp = len(np.intersect1d(p, t))
  fn = len(np.intersect1d(n, f))
  return tp / (tp + fn)
def auc_pr(y_true:np.array, y_pred: np.array):
  t = np.unique(y_pred)
  curve = np.zeros((t.shape[0], 2))
  prec, rec = np.zeros(t.shape[0]), np.zeros(t.shape[0])
  for i in range(t.shape[0]):
    prec[i] = precision(y_true, y_pred >= t[i])
    rec[i] = recall(y_true, y_pred >= t[i])
  curve[:, 0], curve[:, 1] = prec, rec
  return prec.sum()/len(prec)

def train_test(x: np.array, y: np.array, test_size: float=0.3):
  l = int(len(x) * test_size)
  dataset = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
  np.random.shuffle(dataset)
  test, train = dataset[:l], dataset[l:]
  x_train, y_train = train[:, 0], train[:, 1]
  x_test, y_test = test[:, 0], test[:, 1]
  return x_train, x_test, y_train, y_test

def creat_lr():
  # X, y = np.load('X.npy'), np.load('y.npy')
  X_y = np.loadtxt('X and y.txt')
  X, y = X_y[:, 0], X_y[:, 1]

  x_train, x_test, y_train, y_test = train_test(X, y)
  logr = LogisticRegression()
  logr.fit(x_train, y_train, lr=0.1, num_epoch=30)

  # preds = logr.predict(x_test)
  # print(auc_pr(y_test, preds))

  with open('model.pkl', 'wb') as f:
    dill.dump(logr, f)

# creat_lr()