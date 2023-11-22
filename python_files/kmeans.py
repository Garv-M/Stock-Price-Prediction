import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

class FederatedKMeansModel(tf.keras.Model):
    def __init__(self, num_clusters):
        super(FederatedKMeansModel, self).__init__()
        self.num_clusters = num_clusters
        # This is a placeholder; actual KMeans logic will need to be TensorFlow compatible
        self.kmeans = KMeans(n_clusters=self.num_clusters)

    def call(self, inputs):
        # Custom logic to integrate KMeans goes here
	clusters = self.kmeans.fit_predict(inputs)
        # Convert cluster indices to one-hot encoded form for compatibility with TensorFlow
        one_hot_clusters = tf.one_hot(clusters, depth=self.num_clusters)
        return one_hot_clusters
        #return self.kmeans.fit_predict(inputs)

# Load your dataset
df = pd.read_csv('/home/slade/AAPL.csv', usecols=["Open", "High", "Low", "Close"])

# Data preprocessing and client data simulation
# Normalize the data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Function to split the dataset for clients
def split_dataset_for_clients(dataset, num_clients):
    client_data = np.array_split(dataset, num_clients)
    return [tf.data.Dataset.from_tensor_slices((client.iloc[:, :-1].values, client.iloc[:, -1].values)).batch(20).repeat(10) for client in client_data]

# Split the dataset into subsets for each client
num_clients = 5
train_data = split_dataset_for_clients(df_scaled, num_clients)

def model_fn():
    # Create an instance of your custom model
    model = FederatedKMeansModel(num_clusters=3)

    # Dummy input_spec; this should ideally match the structure and data types of your actual dataset
    dummy_input_spec = (tf.TensorSpec(shape=[None, 4], dtype=tf.float32),  # Example for 4 features
                        tf.TensorSpec(shape=[None], dtype=tf.int32))  # Dummy labels

    return tff.learning.models.from_keras_model(
        model,
        input_spec=dummy_input_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(),  # Placeholder loss function
        metrics=[tf.keras.metrics.Accuracy()]  # Placeholder metric
    )


# Federated learning setup
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))  # Optimizer may not be relevant for KMeans
state = trainer.initialize()

# Federated training loop
for _ in range(5):
    result = trainer.next(state, train_data)
    state = result.state
    metrics = result.metrics
    print(metrics['client_work']['train']['loss'])

















