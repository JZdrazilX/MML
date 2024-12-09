import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from typing import Dict, Any, Optional

def generate_color_mapping(
    df: pd.DataFrame,
    column_name: str,
    cmap_name: str = 'viridis',
    rounding: Optional[int] = None
) -> Dict[Any, str]:
    """
    Generate a color mapping for unique values in a DataFrame column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to generate colors for.
        cmap_name (str, optional): The name of the matplotlib colormap to use. Defaults to 'viridis'.
        rounding (Optional[int], optional): Number of decimal places to round the column values.
            If None, no rounding is performed. Defaults to None.

    Returns:
        Dict[Any, str]: A mapping from unique column values to hex color codes.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    if rounding is not None:
        df[column_name] = df[column_name].round(rounding)
    
    unique_values = np.sort(df[column_name].unique())
    num_unique_values = len(unique_values)
    color_palette = plt.cm.get_cmap(cmap_name, num_unique_values)
    colors = [color_palette(i) for i in range(num_unique_values)]
    colors_hex = [mcolors.rgb2hex(color) for color in colors]
    color_map = dict(zip(unique_values, colors_hex))
    
    return color_map

def map_compound_embeddings(
    df: pd.DataFrame,
    compound_column: str = 'Compound'
) -> pd.Series:
    """
    Map compound names to numerical embeddings.

    Args:
        df (pd.DataFrame): The DataFrame containing the compound data.
        compound_column (str, optional): The name of the column with compound names.
            Defaults to 'Compound'.

    Returns:
        pd.Series: A Series with numerical embeddings corresponding to compounds.

    Raises:
        ValueError: If the specified compound column does not exist in the DataFrame.
    """
    if compound_column not in df.columns:
        raise ValueError(f"Column '{compound_column}' does not exist in the DataFrame.")
    
    compound_mapping = {
        '3TFM-2HE': 1,
        '3OMe-3.5DCl': 2,
        '2AD5Cl-3Cl': 3,
        '2AD.5OMe-3Cl': 4,
        '2AD.5OMe-3.5DCl': 5
    }
    compound_emb = df[compound_column].map(compound_mapping)
    if compound_emb.isnull().any():
        missing_compounds = df[compound_column][compound_emb.isnull()].unique()
        raise ValueError(f"Compound embeddings not found for: {missing_compounds}")
    return compound_emb