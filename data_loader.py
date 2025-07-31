import pandas as pd

def load_transactions(path):
    
    """
    Load the transactions CSV into a pandas DataFrame and
    prepare a combined lowercase text field for ML.
    
    Args:
        path (str or file-like): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Original columns plus a new 'text' column.
    """
    # Read CSV into DataFrame
    df = pd.read_csv(path)
    
    # Ensure required columns exist
    expected = {"date", "amount", "merchant", "description"}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {expected}")
    
    # Combine merchant and description into one text field for ML
    df["text"] = (
        df["merchant"].fillna("")    # avoid NaN
        + " "
        + df["description"].fillna("")
    ).str.lower()                   # lowercase for uniformity

    return df
    