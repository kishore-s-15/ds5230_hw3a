import gzip
import struct
from pathlib import Path

import numpy as np
	

class MNISTDataParser:

    # File URLS
    BASE_DIR = Path.cwd().parent
    MNIST_DATA_DIR = BASE_DIR / "data" / "mnist"

    TRAIN_DATA_URL = MNIST_DATA_DIR / "train-images-idx3-ubyte.gz"
    TRAIN_LABELS_URL = MNIST_DATA_DIR / "train-labels-idx1-ubyte.gz"

    TEST_DATA_URL = MNIST_DATA_DIR / "t10k-images-idx3-ubyte.gz"
    TEST_LABELS_URL = MNIST_DATA_DIR / "t10k-labels-idx1-ubyte.gz"


    def load_dataset(self, dataset_url: str) -> np.array:       
        with gzip.open(dataset_url,'rb') as data_file:
            magic, size = struct.unpack(">II", data_file.read(8))
            nrows, ncols = struct.unpack(">II", data_file.read(8))
            data = np.frombuffer(data_file.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, -1))
        
        return data


    def load_labels(self, labels_url: str) -> np.array:
        with gzip.open(labels_url,'rb') as labels_file:
            magic, size = struct.unpack('>II', labels_file.read(8))
            label = np.frombuffer(labels_file.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
            return label


    @staticmethod
    def show_image(image: np.array, title: str) -> None:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap=plt.get_cmap("gray"))
        plt.title(title)
        plt.show()

    
    def parse_data(self, verbose: int = 1, sample_data: bool = False) -> tuple[np.array, np.array, np.array, np.array]:
        X_train = self.load_dataset(self.TRAIN_DATA_URL)
        y_train = self.load_labels(self.TRAIN_LABELS_URL)

        X_test = self.load_dataset(self.TEST_DATA_URL)
        y_test = self.load_labels(self.TEST_LABELS_URL)

        if sample_data:
            n_train_samples = 10000
            train_rand_idx = np.random.choice(X_train.shape[0], n_train_samples, replace=False)
            X_train = X_train[train_rand_idx]
            y_train = y_train[train_rand_idx]

            n_test_samples = 5000
            test_rand_idx = np.random.choice(X_test.shape[0], n_test_samples, replace=False)
            X_test = X_test[test_rand_idx]
            y_test = y_test[test_rand_idx]

        if verbose:
            rand_train_idx = np.random.randint(low=0, high=X_train.shape[0])
            title = f"MNIST Training Image - Label: {y_train[rand_train_idx]}"
            self.show_image(X_train_scaled[rand_train_idx].reshape(28, 28), title)

            rand_test_idx = np.random.randint(low=0, high=X_test.shape[0])
            title = f"MNIST Testing Image - Label: {y_test[rand_test_idx]}"
            self.show_image(X_test_scaled[rand_test_idx].reshape(28, 28), title)

        return (X_train, y_train, X_test, y_test)