# MANUFACTURING DEFECTS PREEDICTOR

This project demonstrates predictive maintenance for industrial machines using a dataset containing operational parameters, maintenance history, and failure information. The goal is to predict machine failures using the Support Vector Classifier (SVC) model.

---

## Dataset Description

### File: `MAINTENANCE PREDICTIVE FOR INDUSTRIAL MACHINES.csv`

This dataset contains 5000 entries and 12 columns:

1. **Machine_ID**: Unique identifier for each machine.
2. **Machine_Type**: Type of machine (e.g., Sealing, Packaging, Filling).
3. **Runtime**: Total runtime of the machine (in hours).
4. **Temperature**: Operating temperature (in °C).
5. **Vibration**: Level of vibration during operation.
6. **Pressure**: Machine pressure during operation.
7. **Power_Consumption**: Power usage of the machine (in watts).
8. **Maintenance_History**: Number of previous maintenance activities performed.
9. **Downtime**: Downtime recorded for the machine (in hours).
10. **Failures**: Binary flag (0 or 1) indicating if a failure occurred.
11. **Remaining_Useful_Life**: Estimated remaining useful life of the machine (in hours).
12. **Ambient_Temperature**: Ambient temperature around the machine (in °C).

---

## Data Analysis and Observations

- The dataset includes operational and environmental parameters, which are essential for identifying patterns leading to machine failures.
- Failure labels (`Failures`) are used as the target variable for classification.
- The dataset does not have any missing values.

### Sample Data

| Machine_ID | Machine_Type | Runtime    | Temperature | Vibration | Pressure  | Power_Consumption | Maintenance_History | Downtime | Failures | Remaining_Useful_Life | Ambient_Temperature |
|------------|--------------|------------|-------------|-----------|-----------|--------------------|----------------------|----------|----------|-----------------------|----------------------|
| 7          | Sealing      | 6566.553173 | 74.883651  | 0.665550  | 38.050919 | 209.748598         | 0                    | 0.0      | 0        | 632.557007            | 35.390333            |
| 4          | Packaging    | 3555.619002 | 72.927617  | 4.168720  | 35.851286 | 428.760901         | 0                    | 0.0      | 1        | 633.456781            | 39.443937            |

---

## Model: Support Vector Classifier (SVC)

The **Support Vector Classifier (SVC)** is used to classify machines into failure (1) or no-failure (0) categories. The model was chosen for its effectiveness in binary classification problems, especially with complex relationships between features.

### Steps to Train and Evaluate the Model:
1. **Data Preprocessing**:
   - Split the dataset into training and testing sets.
   - Scale numerical features using StandardScaler for better SVC performance.

2. **Training**:
   - Train the SVC model using the `Failures` column as the target variable.

3. **Evaluation**:
   - Evaluate the model using metrics like accuracy, precision, recall, and F1-score.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `fastapi`, `uvicorn`

### Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>

   # Machine Failure Prediction API
uvicorn script_name:app --reload
Access the API at http://127.0.0.1:8000.

API Endpoints
/predict
Accepts machine parameters as input and returns the failure prediction.

Example Request:
bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
    "Runtime": 5000,
    "Temperature": 75.0,
    "Vibration": 2.5,
    "Pressure": 40.0,
    "Power_Consumption": 300.0,
    "Ambient_Temperature": 30.0
}


Response:
json

{
    "Machine Failure Prediction": "No Failure"
}



## Contributing
Feel free to open issues or create pull requests for improvements.

## License
This project is licensed under the MIT License.
