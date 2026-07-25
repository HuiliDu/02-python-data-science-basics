import pandas as pd
import numpy as np
import logging
from pathlib import Path


def setup_logging(log_file: Path | None):
    """
    Set up logging configuration.
    If a log file path is provided, logs will be written to"""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise


def introduce_wrong_data(
    df: pd.DataFrame, wrong_data_fraction: float = 0.02
) -> pd.DataFrame:
    """
    Introduce wrong data into the DataFrame by randomly assigning NaN values to a fraction of the rows.
    """
    df_noise = df.copy()
    num_rows = len(df)
    num_wrong_data = int(num_rows * wrong_data_fraction)
    wrong_indices = np.random.choice(df.index, size=num_wrong_data, replace=False)

    nan_indices = wrong_indices[: int(num_wrong_data * 0.5)]
    temperature_indices = wrong_indices[
        int(num_wrong_data * 0.5) : int(num_wrong_data * 0.75)
    ]
    elevation_indices = wrong_indices[int(num_wrong_data * 0.75) :]

    # Assign NaN values to the selected indices for all columns
    df_noise.loc[nan_indices] = np.nan
    logging.info(f"Introduced wrong data (NaN values) to {nan_indices} rows.")
    logging.info(f"Introduced invalid temperature data to {temperature_indices} rows.")
    logging.info(f"Introduced invalid elevation data to {elevation_indices} rows.")

    # Introduce invalid temperature values (e.g., below -10 or above 40) for a fraction of the rows
    low_temperature = np.random.uniform(-50, -10, size=len(temperature_indices))
    high_temperature = np.random.uniform(40, 100, size=len(temperature_indices))
    choice_temperature = np.random.rand(len(temperature_indices)) < 0.5
    df_noise.loc[temperature_indices, "temperature"] = np.where(
        choice_temperature, low_temperature, high_temperature
    )

    # Introduce invalid elevation values (e.g., below 0 or above 3000) for a fraction of the rows
    low_elevation = np.random.uniform(-100, 0, size=len(elevation_indices))
    high_elevation = np.random.uniform(3000, 5000, size=len(elevation_indices))
    choice_elevation = np.random.rand(len(elevation_indices)) < 0.5
    df_noise.loc[elevation_indices, "elevation"] = np.where(
        choice_elevation, low_elevation, high_elevation
    )
    logging.info(
        f"Introduced unrealistic temperature to {len(temperature_indices)} rows and unrealistic elevation to {len(elevation_indices)} rows."
    )
    return df_noise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by handling missing values and invalid values, and ensuring correct data types.
    """
    # Drop rows with any missing values
    df_cleaned = df.dropna().copy()
    logging.info(f"Dropped rows with missing values. Remaining rows: {len(df_cleaned)}")

    # Ensure correct data types
    df_cleaned["latitude"] = df_cleaned["latitude"].astype(float)
    df_cleaned["longitude"] = df_cleaned["longitude"].astype(float)
    df_cleaned["nvdi"] = df_cleaned["nvdi"].astype(float)
    df_cleaned["land_cover"] = df_cleaned["land_cover"].astype(str)
    df_cleaned["pm2.5"] = df_cleaned["pm2.5"].astype(float)
    df_cleaned["rainfall"] = df_cleaned["rainfall"].astype(float)
    df_cleaned["temperature"] = df_cleaned["temperature"].astype(float)
    df_cleaned["elevation"] = df_cleaned["elevation"].astype(float)

    valid_conditions = (
        (df_cleaned["temperature"] <= 40)
        & (df_cleaned["temperature"] >= -10)
        & (df_cleaned["elevation"] >= 0)
        & (df_cleaned["elevation"] <= 3000)
    )

    df_cleaned = df_cleaned[valid_conditions].copy()

    # conditions = (df_cleaned["temperature"] <= 40) & (df_cleaned["temperature"] >= 0)
    # df_cleaned["temperature"] = np.where(
    #     conditions, df_cleaned["temperature"].astype(float), np.nan
    # )
    # conditions = (df_cleaned["elevation"] >= 0) & (df_cleaned["elevation"] <= 3000)
    # df_cleaned["elevation"] = np.where(
    #     conditions, df_cleaned["elevation"].astype(float), np.nan
    # )
    # df_cleaned = df_cleaned.dropna()
    logging.info(f"Dropped rows with invalid values. Remaining rows: {len(df_cleaned)}")
    logging.info(
        f"Cleaned data: {len(df_cleaned)} rows remaining after cleaning missing values.and invalid values."
    )
    logging.info("Data types ensured for all columns.")
    return df_cleaned


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numerical features in the DataFrame using Min-Max scaling.
    """
    numerical_features = [
        "latitude",
        "longitude",
        "temperature",
        "nvdi",
        "elevation",
        "pm2.5",
        "rainfall",
    ]
    for feature in numerical_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        df[feature] = (df[feature] - min_val) / (max_val - min_val)
        logging.info(f"Scaled feature '{feature}' using Min-Max scaling.")

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features in the DataFrame using one-hot encoding.
    """
    categorical_features = ["land_cover"]
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    logging.info(f"Encoded categorical features: {categorical_features}")
    return df_encoded


def save_preprocessed_data(df: pd.DataFrame, output_path: Path):
    """
    Save the preprocessed DataFrame to a CSV file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Preprocessed data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving preprocessed data to {output_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    log_file_path = Path("logs/preprocessing.log")
    setup_logging(log_file_path)

    input_file_path = Path("datasets/synthetic_dataset.csv")
    output_file_path = Path("datasets/preprocessed_dataset.csv")

    df = load_data(input_file_path)
    df_with_wrong_data = introduce_wrong_data(df, wrong_data_fraction=0.05)
    save_preprocessed_data(
        df_with_wrong_data, Path("datasets/dataset_with_wrong_data.csv")
    )
    df_cleaned = clean_data(df_with_wrong_data)
    df_scaled = scale_data(df_cleaned)
    df_encoded = encode_categorical_features(df_scaled)
    save_preprocessed_data(df_encoded, output_file_path)
