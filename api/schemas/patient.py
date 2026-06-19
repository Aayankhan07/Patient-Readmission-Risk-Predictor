from pydantic import BaseModel, Field

class PatientInput(BaseModel):
    """
    Patient demographics, admission details, and lab values.
    All secondary features have defaults matching dataset medians/modes.
    """
    age: str = Field(default="[60-70)", description="Age bracket of the patient (e.g. [50-60), [60-70))")
    time_in_hospital: int = Field(default=3, ge=1, le=14, description="Time spent in hospital in days")
    num_procedures: int = Field(default=1, ge=0, le=6, description="Number of non-lab procedures during encounter")
    num_medications: int = Field(default=15, ge=1, le=81, description="Number of distinct medications prescribed")
    number_diagnoses: int = Field(default=9, ge=1, le=16, description="Number of diagnoses entered in system")
    A1Cresult: str = Field(default="None", description="A1C lab result (e.g. None, Normal, >7, >8)")
    insulin: str = Field(default="No", description="Insulin prescription status (e.g. No, Steady, Up, Down)")
    diabetesMed: str = Field(default="No", description="Whether any diabetic medication was prescribed (Yes, No)")
    
    # Secondary features with defaults
    race: str = "Caucasian"
    gender: str = "Female"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    payer_code: str = "?"
    medical_specialty: str = "?"
    num_lab_procedures: int = 40
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0
    diag_1: str = "250.xx"
    diag_2: str = "428"
    diag_3: str = "276"
    max_glu_serum: str = "None"
    metformin: str = "No"
    repaglinide: str = "No"
    nateglinide: str = "No"
    chlorpropamide: str = "No"
    glimepiride: str = "No"
    acetohexamide: str = "No"
    glipizide: str = "No"
    glyburide: str = "No"
    tolbutamide: str = "No"
    pioglitazone: str = "No"
    rosiglitazone: str = "No"
    acarbose: str = "No"
    miglitol: str = "No"
    troglitazone: str = "No"
    tolazamide: str = "No"
    examide: str = "No"
    citoglipton: str = "No"
    glyburide_metformin: str = Field(default="No", alias="glyburide-metformin")
    glipizide_metformin: str = Field(default="No", alias="glipizide-metformin")
    glimepiride_pioglitazone: str = Field(default="No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: str = Field(default="No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: str = Field(default="No", alias="metformin-pioglitazone")
    change: str = "No"

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": "[60-70)",
                "time_in_hospital": 5,
                "num_procedures": 2,
                "num_medications": 14,
                "number_diagnoses": 7,
                "A1Cresult": ">8",
                "insulin": "Steady",
                "diabetesMed": "Yes"
            }
        }
