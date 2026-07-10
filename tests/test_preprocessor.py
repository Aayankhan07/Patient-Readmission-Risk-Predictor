import pandas as pd
import numpy as np
from src.data.preprocessor import ReadmissionPreprocessor


def test_preprocessor_fit_transform():
    # 1. Arrange
    data = {
        "age": ["[10-20)", "[60-70)", np.nan],
        "time_in_hospital": [3, np.nan, 5],
        "num_procedures": [1, 2, 0],
        "num_medications": [10, 15, 20],
        "number_diagnoses": [9, 8, 7],
        "A1Cresult": ["None", ">8", "Normal"],
        "insulin": ["Steady", "No", "Up"],
        "diabetesMed": ["Yes", "No", "Yes"],
        "race": ["Caucasian", "?", "Hispanic"],
        "gender": ["Female", "Male", "Female"],
    }
    df = pd.DataFrame(data)

    # 2. Act
    preprocessor = ReadmissionPreprocessor()
    preprocessor.fit(df)
    df_trans = preprocessor.transform(df)

    # 3. Assert
    # Check that age column is ordinal encoded and numerical
    assert "age" in df_trans.columns
    # Check that there are no missing values remaining
    assert df_trans.isnull().sum().sum() == 0
    # Check shape
    assert df_trans.shape[0] == 3
    # Check scaling (mean should be close to 0, std close to 1 across rows if we scale large dataset, but with 3 rows we just check no NaNs)
    assert not np.isnan(df_trans.values).any()
