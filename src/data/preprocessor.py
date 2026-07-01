import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.utils.config import PREPROCESSOR_PATH

class ReadmissionPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor class for the Patient Readmission Risk Predictor.
    Handles imputation, ordinal encoding of age, one-hot encoding, and scaling.
    """
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.median_values = {}
        self.mode_values = {}
        self.scaler = StandardScaler()
        self.feature_columns_ = None
        self.age_mapping = {
            "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
            "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
        }
        
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        
        # Replace '?' with NaN
        X = X.replace('?', np.nan)
        
        # Identify numeric and categorical columns
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Avoid treating target or ID columns in features if they exist
        for col in ["encounter_id", "patient_nbr", "readmitted"]:
            if col in self.numeric_cols:
                self.numeric_cols.remove(col)
            if col in self.categorical_cols:
                self.categorical_cols.remove(col)
                
        # Fit median and mode imputers
        for col in self.numeric_cols:
            self.median_values[col] = X[col].median()
            if pd.isna(self.median_values[col]):
                self.median_values[col] = 0.0
                
        for col in self.categorical_cols:
            if col == "age":
                continue
            self.mode_values[col] = X[col].mode().iloc[0] if not X[col].mode().empty else "None"
            
        # Transform temporarily to fit scaler
        X_trans = self._preprocess_except_scaling(X)
        self.feature_columns_ = X_trans.columns.tolist()
        self.scaler.fit(X_trans)
        return self

    def _preprocess_except_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.replace('?', np.nan)
        
        # Impute numeric
        for col in self.numeric_cols:
            val = self.median_values.get(col, 0.0)
            X[col] = X[col].fillna(val)
            
        # Impute categorical
        for col in self.categorical_cols:
            if col == "age":
                continue
            val = self.mode_values.get(col, "None")
            X[col] = X[col].fillna(val)
            
        # Age Ordinal Encoding
        X['age'] = X['age'].fillna("[50-60)")  # default to middle bracket if NaN
        X['age'] = X['age'].map(lambda x: self.age_mapping.get(str(x), 5))
        
        # One-hot encoding for nominal categories
        nominal_cols = [c for c in self.categorical_cols if c != "age"]
        X_encoded = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
        
        # Ensure all columns exist as expected (handling unseen or missing categories at test/inference time)
        if self.feature_columns_ is not None:
            # Reindex with fitted feature columns, filling missing with 0 (since they are one-hot)
            # Only keep columns in feature_columns_
            missing_cols = set(self.feature_columns_) - set(X_encoded.columns)
            for c in missing_cols:
                X_encoded[c] = 0
            X_encoded = X_encoded[self.feature_columns_]
        
        # Drop ID features if they are still there
        for col in ["encounter_id", "patient_nbr", "readmitted"]:
            if col in X_encoded.columns:
                X_encoded = X_encoded.drop(columns=[col])
                
        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_trans = self._preprocess_except_scaling(X)
        scaled_data = self.scaler.transform(X_trans)
        return pd.DataFrame(scaled_data, columns=self.feature_columns_)

def preprocess_raw_dataset(df: pd.DataFrame):
    """Processes the full dataset, fits preprocessor, splits and applies SMOTE."""
    df = df.copy()
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # 1. Drop columns with > 40% missing values
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.40].index.tolist()
    print(f"Dropping columns with > 40% missing values: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    
    # 2. Drop duplicate rows
    df = df.drop_duplicates()
    
    # 3. Map target: readmitted -> binary (1 if <30, else 0)
    if "readmitted" in df.columns:
        df["readmitted"] = df["readmitted"].apply(lambda x: 1 if str(x).strip() == "<30" else 0)
    
    return df

def apply_smote(X: pd.DataFrame, y: pd.Series):
    """Balances classes in the training set using SMOTE."""
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.splitter import split_data
    # Import the class from the module namespace so that pickling works correctly
    from src.data.preprocessor import ReadmissionPreprocessor
    
    print("Running data preprocessing stage...")
    # 1. Load Raw Dataset
    df = load_raw_data()
    
    # 2. Preprocess Raw Data (drop missing, mapping target)
    df_clean = preprocess_raw_dataset(df)
    
    # 3. Stratified Split 70/15/15
    df_train, df_val, df_test = split_data(df_clean)
    
    # Separate features and target
    X_train_raw = df_train.drop(columns=["readmitted"])
    y_train = df_train["readmitted"]
    X_val_raw = df_val.drop(columns=["readmitted"])
    y_val = df_val["readmitted"]
    X_test_raw = df_test.drop(columns=["readmitted"])
    y_test = df_test["readmitted"]
    
    # 4. Fit Preprocessor on Train set and transform all
    preprocessor = ReadmissionPreprocessor()
    preprocessor.fit(X_train_raw)
    
    X_train_scaled = preprocessor.transform(X_train_raw)
    X_val_scaled = preprocessor.transform(X_val_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)
    
    # Save the preprocessor to models/preprocessor.pkl
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved fitted preprocessor to {PREPROCESSOR_PATH}")
    
    # 5. Apply SMOTE to training set only
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)
    
    # Ensure test arrays are numeric for models
    X_train_res = pd.DataFrame(X_train_res, columns=preprocessor.feature_columns_)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=preprocessor.feature_columns_)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=preprocessor.feature_columns_)
    
    # Save processed splits locally
    processed_dir = os.path.dirname(PREPROCESSOR_PATH).replace("models", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    X_train_res.assign(readmitted=y_train_res).to_csv(os.path.join(processed_dir, "train_processed.csv"), index=False)
    X_val_scaled.assign(readmitted=y_val).to_csv(os.path.join(processed_dir, "val_processed.csv"), index=False)
    X_test_scaled.assign(readmitted=y_test).to_csv(os.path.join(processed_dir, "test_processed.csv"), index=False)
    print("Preprocessing complete. Processed splits saved in data/processed/.")

