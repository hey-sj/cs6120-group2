# 1. Description

This project will attempt to predict intensive care unit (ICU) length of stay (LOS) using emergency department (ED) data for patients admitted to an ICU from the ED. We will be using the Medical Information Mart for Intensive Care (MIMIC)-IV database, a relational database comprised of deidentified electronic health records (EHRs) for patients admitted to Beth Israel Deaconess Medical Center in Boston, MA, USA. We will be using the `icu` and the `ed` modules. 

# 2. Dataset

This project uses MIMIC-IV, a public database containing real hospital stays for patients. Specifically, we use modules as follows: 

- **ICU Module**: Contains information collected from the clinical information system used within the ICU. 
    - **icustays_cleaned.csv**: Cleaned tracking table for ICU stays. 
        - `ham_id`: A unique hospital identifier (ranges from 2000000 - 2999999) used to link the ED stay with the hospitalization in MIMIC-IV. If hadm_id is NULL, the patient was not admitted to the hospital after their ED stay.
        - `los`: The length of stay for the patient for the given ICU stay, which may include one or more ICU units. The length of stay is measured in fractional days.

- **ED Module**: Contains data for emergency department patients collected while they are in the ED. 
    - **edstays_cleaned.csv**: Cleaned tracking table for ED visits. 
        - `subject_id`: A unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual. 
        - `stay_id`: An identifier which uniquely identifies a single emergency department stay for a single patient. 
        - `ham_id`: A unique hospital identifier (ranges from 2000000 - 2999999) used to link the ED stay with the hospitalization in MIMIC-IV. If hadm_id is NULL, the patient was not admitted to the hospital after their ED stay.
        - `intime`, `outtime`: The admission datetime (intime) and discharge datetime (outtime) of the given emergency department stay. 
    - **vitals_cleaned.csv**: Cleaned table for time-stamped vital signs. 
        - `stay_id`: An identifier which uniquely identifies a single emergency department stay for a single patient. 
        - `heartrate`: The patient's heart rate in beats per minute.
        - `resprate`: The patient's respiratory rate in breaths per minute.
        - `o2sat`: The patient's oxygen saturation measured as a percentage.
        - `sbp`, `dbp`: The patient's systolic (sbp) and diastolic (dbp) blood pressure measured in millimetres of mercury (mmHg).
    - **triage_cleaned.csv**: Cleaned table for initial triage measurements in ED. 
        - `stay_id`: An identifier which uniquely identifies a single emergency department stay for a single patient. 
        - `heartrate`: The patient's heart rate in beats per minute.
        - `resprate`: The patient's respiratory rate in breaths per minute.
        - `o2sat`: The patient's oxygen saturation measured as a percentage.
        - `sbp`, `dbp`: The patient's systolic (sbp) and diastolic (dbp) blood pressure measured in millimetres of mercury (mmHg).
    - **diagnosis_cleaned.csv**: Cleaned table for diagnosis codes and descriptions. 
        - `icd_code`: A coded diagnosis using the International Classification of Diseases (ICD) ontology.
        - `icd_version`: The version of the ICD system used; either 9 indicating ICD-9 or 10 indicating ICD-10. The ontologies for these two systems differ, and therefore the meaning of the icd_code will depend on the icd_version.

Data Preparation:

1. Cleaned `edstays` table under MIMIC-IV-ED to remove ED stays that were not admitted later into the hospital - e.g. only include rows with an `hadm_id`. 
2. Cleaned `icustays` table under MIMIC-IV-ICU to only include rows with an `hadm_id` that is also in `edstays` table.
3. Update edstays table to only include rows with an `hadm_id` that is also in `icustays` - this allows us to limit the `edstays` table so that we can observe emergency department stays that were admitted to the critical care units.
4. Clean `triage` table under MIMIC-IV-ED to only include stays that are in the updated `edstays` table.
5. Clean `vitalsign` table under MIMIC-IV-ED to only include stays that are in the updated `edstays` table.
6. Clean `diagnosis` table under MIMIC-IV-ED to only include stays that are in the updated `edstays` table.

### Number of Rows in Each Table
| Table              | Before Cleaning | After Cleaning |
| ---------------- | ------ | ---- |
| edstays       | 425087 | 31916 |
| icustays      |   73181   | 35179 |
| triage        |  425087   | 31916 |
| vitalsign |  1564610   | 198919 |
| diagnosis | 899050 | 68254 |

We have included the cleaned tables in this repository under the respective `ed` and `icu` folders. This is **prior to preprocessing**, which we describe below.

# 3. Framework

![figure](https://github.com/user-attachments/assets/a8ec38ec-a66a-4392-8b02-93c2dd65d9ed)

**<h3>Preprocessing</h3>**

- **Combining ICD Codes**:
The `icd_version` and `icd_code` fields are combined into `icd_combined` to standardize diagnosis information across different coding systems. 
- **Merging ED features with ICU Data**: The ED and ICU datasets are merged using the patient's `stay_id` and `hadm_id`, making sure that each row in the merged dataset represents a unique patient admission.
- **Imputing Missing Values and removing outliers**: Missing values in vital signs and other critical fields are imputed using mean values and outliers are removed to ensure the dataset is complete and proper for analysis.

**<h3>Feature Engineering</h3>**

- **Static Features**: Mean values of vital signs (`heartrate`, `resprate`, `o2sat`, `sbp`/`dbp`) are calculated for each ED stay, which represent the patient's overall condition during their ED visit.
- **Dynamic Features**: Linear regression is used to compute the slope of each vital sign over time, indicating whether the patient's health condition is improving or deteriorating.
- **Combining Features**: Static and dynamic features are merged together to ensure that both overall and detailed patterns in patient health are included.

**<h3>Modeling</h3>**

- **Linear Regression**: A baseline model to see initial performance.
- **Random Forest**: A non-linear ensemble model that handles complex interactions between variables effectively.
- **Long Short-Term Memory(LSTM)**: A type of recurrent neural network designed to handle sequential data, such as time-series LOS trends. 
- **Combined Model**: Random forest and LSTM predictions are combined to leverage the strengths of both models. The random forest handles non-linear relationships, while LSTM handles time-dependent data. 
- **Training and Validation**: The dataset is split into 80% training data and 20% validation data.

**<h3>Evaluation</h3>**
- **Performance Metrics**: Mean squared error (MSE) is used to measure the average squared difference between actual and predicted ICU LOS, and lower MSE indicates better predictive accuracy.
R-squared is used to represent the proportion of variance in ICU LOS that the model explains, and a value closer to 1 indicates a better fit. 
- **Findings**: Random Forest achieved the lowest MSE and highest R-squared, outperforming all other models. The combined model performed the second best. 


# 4. Results

Of the regression models, random forest outperformed linear regression. It also outperformed the LSTM model and the combined random forest and LSTM model. The dataset we ultimately worked with is about 75000 rows - which may be insufficient for a deep learning model. A larger dataset may further benefit deep learning models like LSTM. In addition, using time-based trends along with static information can make predictions more accurate. This project provides some insight on data size, type, and processing to solving the ICU LOS prediction problem using ED data with machine learning approaches. 


# 5. Pre-requisites

**<h3>Language</h3>**
- Python 3.6

**<h3>Libraries</h3>**
- Sklearn 11.5
- pytorch 10.8
- pandas
- numpy

**<h3>Dataset</h3>**
- Make sure data files are in the expected directories

# 6. How to Run the Code

Run with `python3 icu-los.py` in the terminal, or the equivalent of running a python script for the user's system (e.g. `python icu-los.py`).
