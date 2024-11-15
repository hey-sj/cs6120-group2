Introduction

Examples

How to Run and Hardware Requirements

# Predicting ICU LOS Using ED Data

This project will attempt to predict intensive care unit (ICU) length of stay (LOS) using emergency department (ED) data for patients admitted to an ICU from the ED. We will be using the Medical Information Mart for Intensive Care (MIMIC)-IV database, a relational database comprised of deidentified electronic health records (EHRs) for patients admitted to Beth Israel Deaconess Medical Center in Boston, MA, USA. We will be using the `icu` and the `ed` modules. 

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

We have included the cleaned tables in this repository under the respective `ed` and `icu` folders. 
