# data labeled
# small dataset
# data is noise free
import numpy as np
from collections import Counter
def euclidean_distance(x1, x2):
	return np.sqrt(np.sum((x1 - x2)**2))


class KNN:
	def __init__(self, k:int = 3):
		self.k = k

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		predictions = [self._predict(x) for x in X]
		return predictions

	def _predict(self, x):
		# compute the distance
		distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

		# get the closest k
		k_indices = np.argsort(distances)[:self.k] # sorted in ascending order
		k_nearest_labels = [self.y_train[i] for i in k_indices]

		# majority vote
		most_common = Counter(k_nearest_labels).most_common()
		return most_common[0][0]

if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 7
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions)
    print("KNN classificat7ion accuracy", accuracy(y_test, predictions))