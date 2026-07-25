from pathlib import Path
import pandas as pd
import numpy as np

from DataGenerate.generate_datasets import (
    generate_datasets,
)  # first generate_datasets is the module, second generate_datasets is the function defined in the module


def test_generate_datasets():
    """
    Test the generate_datasets function to ensure it creates a DataFrame with the expected structure and values.
    """
    num_samples = 1000
    df = generate_datasets(num_samples=num_samples)

    # Check if the DataFrame has the correct number of samples
    assert len(df) == num_samples, f"Expected {num_samples} samples, but got {len(df)}"

    # Check if all expected columns are present in the DataFrame
    expected_columns = [
        "latitude",
        "longitude",
        "temperature",
        "nvdi",
        "elevation",
        "land_cover",
        "pm2.5",
        "rainfall",
    ]
    for column in expected_columns:
        assert column in df.columns, f"Missing expected column: {column}"

    # Check if land_cover values are within the expected set
    valid_land_cover_values = {"forest", "urban", "agriculture", "water"}
    assert set(df["land_cover"].unique()).issubset(valid_land_cover_values)
