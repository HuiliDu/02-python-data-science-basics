from pathlib import Path
import pandas as pd
import numpy as np


def generate_datasets(num_samples=1000):
    """
    Generate synthetic datasets with random values for various features.
    """
    np.random.seed(42)  # For reproducibility

    # Generate synthetic data dictionary with random values for each feature
    data = {
        "latitude": np.random.uniform(39.5, 40.5, num_samples),
        "longitude": np.random.uniform(113.5, 114.5, num_samples),
        "temperature": np.random.uniform(-10, 40, num_samples),
        "nvdi": np.random.uniform(-1, 1, num_samples),
        "elevation": np.random.uniform(0, 3000, num_samples),
        "land_cover": np.random.choice(["forest", "urban", "water"], num_samples),
        "pm2.5": np.random.uniform(0, 150, num_samples),
        "rainfall": np.random.uniform(0, 200, num_samples),
    }

    # Create a DataFrame(a pandas structure) from the synthetic data
    df = pd.DataFrame(data)
    # Assign land cover based on nvdi values
    conditions = [df["nvdi"] > 0.6, df["nvdi"] < 0]
    # Define the corresponding land cover types in the same order as the conditions
    choices = ["forest", "water"]
    # Use np.select to assign land cover based on the conditions and choices, with a default value of "urban"
    df["land_cover"] = np.select(conditions, choices, default="urban")
    return df


def save_datasets(df, output_dir="datasets"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "synthetic_dataset.csv", index=False)
    print(f"Dataset saved to {output_path / 'synthetic_dataset.csv'}")


if __name__ == "__main__":
    df = generate_datasets(num_samples=1000)
    save_datasets(df, output_dir="datasets")
