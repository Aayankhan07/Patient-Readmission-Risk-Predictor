import pandas as pd
from sklearn.model_selection import train_split = None
from sklearn.model_selection import train_test_split
from src.utils.config import PARAMS

def split_data(df: pd.DataFrame, target_col: str = "readmitted"):
    """
    Splits the dataframe into stratified train, validation, and test sets.
    Ratios: 70% Train, 15% Validation, 15% Test.
    """
    test_size = PARAMS["data"]["test_size"]
    val_size = PARAMS["data"]["val_size"]
    random_state = PARAMS["data"]["random_state"]
    
    # Stratified split to separate test set (15%)
    df_train_val, df_test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target_col] if target_col in df.columns else None,
        random_state=random_state
    )
    
    # Calculate relative validation size for the remaining 85% of data
    # val_size / (1 - test_size) = 0.15 / 0.85 = 0.17647
    relative_val_size = val_size / (1.0 - test_size)
    
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=relative_val_size,
        stratify=df_train_val[target_col] if target_col in df_train_val.columns else None,
        random_state=random_state
    )
    
    print(f"Dataset split complete:")
    print(f"  Train: {df_train.shape[0]} rows ({df_train.shape[0]/df.shape[0]:.1%})")
    print(f"  Val:   {df_val.shape[0]} rows ({df_val.shape[0]/df.shape[0]:.1%})")
    print(f"  Test:  {df_test.shape[0]} rows ({df_test.shape[0]/df.shape[0]:.1%})")
    
    return df_train, df_val, df_test
