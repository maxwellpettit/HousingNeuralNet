import math
import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# Set logging levels and global settings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


def parse_data():
    # Read sample CSV data.
    california_housing_dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
        sep=",")

    # Randomize the data.
    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))

    # Choose the first 12000 (out of 17000) examples for training.
    training_examples = preprocess_features(
        california_housing_dataframe.head(12000))
    training_targets = preprocess_targets(
        california_housing_dataframe.head(12000))

    # Choose the last 5000 (out of 17000) examples for validation.
    validation_examples = preprocess_features(
        california_housing_dataframe.tail(5000))
    validation_targets = preprocess_targets(
        california_housing_dataframe.tail(5000))

    normalized_dataframe = normalize(
        preprocess_features(california_housing_dataframe))
    normalized_training_examples = normalized_dataframe.head(12000)
    normalized_validation_examples = normalized_dataframe.tail(5000)

    # Non-normalized regression model
    # return training_examples, training_targets, validation_examples, validation_targets

    # Mixed-normalized regression model
    return normalized_training_examples, training_targets, normalized_validation_examples, validation_targets


def preprocess_features(dataframe):
    """Prepares input features from a data set.

    Args:
        dataframe: A Pandas DataFrame expected to contain a dataset.
    Returns:
        A DataFrame that contains the features to be used for the model, including
        synthetic features.
    """
    selected_features = dataframe[[
        "latitude", "longitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income"
    ]]
    processed_features = selected_features.copy()

    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
        dataframe["total_rooms"] / dataframe["population"])
    processed_features["income_per_room"] = (
        dataframe["median_income"] / dataframe["total_rooms"])

    return processed_features


def preprocess_targets(dataframe):
    """Prepares target features (i.e., labels) from a data set.

    Args:
        dataframe: A Pandas DataFrame expected to contain data
        from a data set.
    Returns:
        A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
        dataframe["median_house_value"] / 1000.0)
    return output_targets


def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
        input_features: The names of the numerical input features to use.
    Returns:
        A set of feature columns
    """
    latitude = tf.feature_column.numeric_column("latitude")
    longitude = tf.feature_column.numeric_column("longitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    total_rooms = tf.feature_column.numeric_column("total_rooms")
    total_bedrooms = tf.feature_column.numeric_column("total_bedrooms")
    population = tf.feature_column.numeric_column("population")
    households = tf.feature_column.numeric_column("households")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
    income_per_room = tf.feature_column.numeric_column("income_per_room")

    # Divide longitude into 10 buckets.
    bucketized_longitude = tf.feature_column.bucketized_column(
        longitude,
        boundaries=get_fixed_boundaries(-124.35, -114.31, 20))

    # Divide latitude into 10 buckets.
    bucketized_latitude = tf.feature_column.bucketized_column(
        latitude,
        boundaries=get_fixed_boundaries(32.54, 41.95, 20))

    # Make a feature column for the long_x_lat feature cross
    long_x_lat = tf.feature_column.embedding_column(
        tf.feature_column.crossed_column(
            set([bucketized_longitude, bucketized_latitude]),
            hash_bucket_size=400), 20)

    feature_columns = set([
        latitude, longitude, housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income, rooms_per_person,
        income_per_room, long_x_lat
    ])

    return feature_columns


def get_quantile_based_boundaries(feature_values, num_buckets):
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]


def get_linear_boundaries(feature_values, num_buckets):
    min_val = feature_values.min()
    max_val = feature_values.max()
    boundaries = np.linspace(
        start=min_val, stop=max_val, num=num_buckets, endpoint=False).tolist()
    del boundaries[0]
    return boundaries


def get_fixed_boundaries(min_val, max_val, num_buckets):
    boundaries = np.linspace(
        start=min_val, stop=max_val, num=num_buckets, endpoint=False).tolist()
    del boundaries[0]
    return boundaries


def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)


def log_normalize(series):
    return series.apply(lambda x: math.log(x + 1.0))


def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x: min(max(x, clip_to_min), clip_to_max))


def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x: (x - mean) / std_dv)


def binary_threshold(series, threshold):
    return series.apply(lambda x: (1 if x > threshold else 0))


def normalize(dataframe):
    """Returns a version of the input `DataFrame` that has all its features normalized.

    households, median_income and total_bedrooms all appear normally-distributed in a log space.

    latitude, longitude and housing_median_age would probably be better off just scaled linearly,
    as before.

    population, totalRooms and rooms_per_person have a few extreme outliers. They seem too extreme
    for log normalization to help. So let's clip them instead.
    """
    processed_features = pd.DataFrame()

    processed_features["households"] = log_normalize(dataframe["households"])
    processed_features["median_income"] = log_normalize(
        dataframe["median_income"])
    processed_features["total_bedrooms"] = log_normalize(
        dataframe["total_bedrooms"])

    processed_features["latitude"] = dataframe["latitude"]
    processed_features["longitude"] = dataframe["longitude"]

    processed_features["housing_median_age"] = linear_scale(
        dataframe["housing_median_age"])

    processed_features["population"] = linear_scale(
        clip(dataframe["population"], 0, 5000))
    processed_features["rooms_per_person"] = linear_scale(
        clip(dataframe["rooms_per_person"], 0, 5))
    processed_features["total_rooms"] = linear_scale(
        clip(dataframe["total_rooms"], 0, 10000))

    processed_features["income_per_room"] = linear_scale(
        dataframe["income_per_room"])

    return processed_features


def create_dnn_regressor(feature_columns, model_dir, optimizer, hidden_units):
    run_config = tf.contrib.learn.RunConfig(
        model_dir=model_dir, keep_checkpoint_max=1)

    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=run_config)
    return dnn_regressor


def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural network model.

    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: True or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
        Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # 'features' is a dictionary in which each value is a batch of values for that feature.
    # 'labels' is a batch of labels.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_regression_model(
        optimizer, steps, batch_size, hidden_units, training_examples,
        training_targets, validation_examples, validation_targets, model_dir):
    """Trains a neural network regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
        optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
        steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        hidden_units: A `list` of int values, specifying the number of neurons in each layer.
        training_examples: A `DataFrame` containing one or more columns to use as input
        features for training.
        training_targets: A `DataFrame` containing exactly one column to use as target
        for training.
        validation_examples: A `DataFrame` containing one or more columns to use as
        input features for validation.
        validation_targets: A `DataFrame` containing exactly one column to use as
        target for validation.

    Returns:
        dnn_regressor: the trained `DNNRegressor` object.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a DNNRegressor object.
    feature_columns = construct_feature_columns(training_examples)
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    dnn_regressor = create_dnn_regressor(feature_columns, model_dir, optimizer,
                                         hidden_units)

    # Create input functions.
    training_input_fn = lambda: input_fn(training_examples, training_targets["median_house_value"], batch_size=batch_size)

    predict_training_input_fn = lambda: input_fn(training_examples, training_targets["median_house_value"], num_epochs=1, shuffle=False)

    predict_validation_input_fn = lambda: input_fn(validation_examples, validation_targets["median_house_value"], num_epochs=1, shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess loss metrics.
    print("Training model...")
    print("RMSE (on training data):")

    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(
            input_fn=predict_training_input_fn)
        training_predictions = np.array(
            [item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(
            input_fn=predict_validation_input_fn)
        validation_predictions = np.array(
            [item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_rmse = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_rmse = math.sqrt(
            metrics.mean_squared_error(validation_predictions,
                                       validation_targets))

        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_rmse))

        # Add the loss metrics from this period to our list.
        training_errors.append(training_rmse)
        validation_errors.append(validation_rmse)

    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plot_results(training_errors, validation_errors)

    print("Final RMSE (on training data):   %0.2f" % training_rmse)
    print("Final RMSE (on validation data): %0.2f" % validation_rmse)

    dnn_regressor.export_savedmodel('exports', export_input_fn)

    return dnn_regressor


def load_nn_regression_model(optimizer, hidden_units, training_examples,
                             model_dir):
    feature_columns = construct_feature_columns(training_examples)
    dnn_regressor = create_dnn_regressor(feature_columns, model_dir, optimizer,
                                         hidden_units)

    return dnn_regressor


def get_prediction(dnn_regressor, validation_examples, validation_targets):
    # Create input function.
    predict_validation_input_fn = lambda: input_fn(validation_examples, validation_targets["median_house_value"], num_epochs=1, shuffle=False)

    # Compute predictions.
    validation_predictions = dnn_regressor.predict(
        input_fn=predict_validation_input_fn)
    validation_predictions = np.array(
        [item['predictions'][0] for item in validation_predictions])

    print(str(validation_predictions.size))

    # Compute validation loss.
    validation_rmse = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))

    print("Predicted RMSE: %0.2f" % validation_rmse)
    return validation_rmse


def plot_results(training_rmse, validation_rmse):
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    #weights0 = dnn_regressor.get_variable_value('dnn/hiddenlayer_0/kernel')
    #plt.imshow(weights0, cmap='gray')
    #plt.show()

    #weights1 = dnn_regressor.get_variable_value('dnn/hiddenlayer_1/kernel')
    #plt.imshow(weights1, cmap='gray')
    #plt.show()


def main():
    # Parse whether to train or predict
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        help=
        "'t' to train model or 'p' to predict using previously trained model")
    args = parser.parse_args()
    action = args.action

    # Parse sample CSV data.
    training_examples, training_targets, \
    validation_examples, validation_targets = parse_data()

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.15)
    hidden_units = [10, 10]
    model_dir = 'model/normalized'

    if (action == 't'):
        # Train model.
        dnn_regressor = train_nn_regression_model(
            optimizer=optimizer,
            steps=1000,
            batch_size=100,
            hidden_units=hidden_units,
            training_examples=training_examples,
            training_targets=training_targets,
            validation_examples=validation_examples,
            validation_targets=validation_targets,
            model_dir=model_dir)

    elif (action == 'p'):
        # Get prediction from previously trained model.
        dnn_regressor = load_nn_regression_model(
            optimizer=optimizer,
            hidden_units=hidden_units,
            training_examples=training_examples,
            model_dir=model_dir)

        _ = get_prediction(
            dnn_regressor,
            validation_examples=validation_examples,
            validation_targets=validation_targets)


if __name__ == '__main__':
    main()
