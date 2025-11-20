# Data Governance and Compliance

This repository uses multiple **public, de-identified medical imaging datasets**
for research into adversarially robust and explainable deep learning for
medical imaging. This document explains:

- which datasets are used and under what terms
- how data access and provenance are logged
- how compliance checks are performed in code
- how data lineage and versioning are handled (DVC + Git)
- how the setup aligns with GDPR/HIPAA-style privacy principles

This document is **descriptive**, not legal advice. For any real deployment,
institutional ethics and legal teams must be consulted.

---

## 1. Supported Datasets and Terms (High Level)

### 1.1 ISIC 2018 / 2019 / 2020 Dermoscopy

- **Type**: Dermoscopy images of skin lesions.
- **Primary use**: Algorithm development and research into melanoma detection.
- **Source**:
  - ISIC 2018: `https://challenge2018.isic-archive.com/`
  - ISIC 2019: `https://challenge2019.isic-archive.com/`
  - ISIC 2020: `https://challenge2020.isic-archive.com/`
- **Terms (summary)**:
  - Public challenge / archive datasets for research and education.
  - No direct clinical/diagnostic use.
  - Follow ISIC Archive terms of use and challenge rules.
  - Cite the challenge organisers and associated papers.
- **Identifiers/PHI**:
  - Images and metadata are de-identified; no names, IDs, or contact data.

### 1.2 Derm7pt

- **Type**: Dermoscopy images annotated with 7-point criteria + diagnosis.
- **Source**: `https://github.com/jeremykawahara/derm7pt`
- **Terms (summary)**:
  - Research-oriented dataset; users must comply with the license specified
    in the repository and cite the associated publication.
  - Not intended for standalone clinical decision making.
- **Identifiers/PHI**:
  - Images are de-identified; no direct personal identifiers.

### 1.3 NIH ChestXray14 (NIH CXR)

- **Type**: 100k+ posterior-anterior chest radiographs with multi-label disease
  annotations.
- **Source**: NIH Clinical Center (ChestXray14 dataset).
- **Access**:
  - Distributed via NIH cloud storage after accepting a Data Use Agreement.
- **Terms (summary)**:
  - De-identified radiographs released for scientific research.
  - Users must agree not to attempt re-identification of patients.
  - Not intended for clinical diagnosis; no warranty of fitness for medical use.
- **Identifiers/PHI**:
  - Images are de-identified; the dataset documentation states that PHI has been
    removed. Only derived labels and pseudo-identifiers are used.

### 1.4 PadChest

- **Type**: Chest radiographs with detailed, multi-label annotations and text.
- **Source**: BIMCV / PadChest project (`PadChest` database).
- **Access**:
  - Download typically requires a signed data usage agreement.
- **Terms (summary)**:
  - Released for scientific research and education.
  - Redistribution or sale of the data to third parties is **forbidden**.
  - Dataset is **not** intended for diagnosis or clinical decision making.
  - Users must not attempt to re-identify subjects or share raw data.
- **Identifiers/PHI**:
  - Data are de-identified; only radiographs and derived annotations are used.

### 1.5 General Usage Policy for This Project

Across all datasets, this project adopts the most conservative interpretation:

- **Purpose**: Non-commercial academic research and education only.
- **No clinical deployment**:
  - Models trained here are prototypes, not medical devices.
  - They must **not** be used to make or support clinical decisions.
- **No re-identification**:
  - No attempts are made to re-identify individuals.
  - No linking with other datasets for re-identification purposes.
- **Storage**:
  - Data are stored on local research machines under institutional policies.
  - No public redistribution of raw images or metadata.

---

## 2. Data Governance Module (`src/datasets/data_governance.py`)

The `data_governance` module provides a small, focused Python API for:

- logging **data access** events
- logging **provenance** (which inputs produced which outputs)
- performing basic **compliance checks**
- exposing **dataset metadata** (license/terms summaries)

### 2.1 Dataset Metadata

- `DatasetInfo` and `DatasetLicenseInfo` store:
  - canonical key (`isic2018`, `nih_cxr`, `padchest`, `derm7pt`, …)
  - display name and source URL
  - short license/terms summary
  - allowed purposes (`research`, `education`)
  - whether commercial use is allowed (here: `False` for all datasets)
  - whether datasets contain direct identifiers (here: `False`)

Example:

```python
from src.datasets import data_governance as gov

info = gov.get_dataset_info("isic2020")
print(info.display_name)         # "ISIC 2020 Dermoscopy"
print(info.license.summary)      # human-readable license summary
