# Fair Speech Emotion Recognition

## Project Overview
This project investigates **fairness issues in speech emotion recognition systems**. It demonstrates how bias can manifest in machine learning models and implements techniques to mitigate this bias, focusing on **gender fairness** in emotion classification.

---

## Academic Context
- **Author:** Laura Capella
- **Course:** Logics for AI – HCAI (2024–2025)  
- **Professor:** Giuseppe Primiero  
- **Module 3 Topics:**  
  Data, Data Science, Data Quality, Bias and Bias Mitigation, Trustworthiness

---

## Key Features
- Extracts acoustic features (MFCCs) from speech audio using **librosa**
- Intentionally induces **gender bias** in the dataset for demonstration
- Implements and compares two **fairness mitigation techniques**:
  - Prejudice Remover (in-processing)
  - Reweighing (pre-processing)
- Evaluates models using:
  - **Performance metrics:** Accuracy, ROC AUC
  - **Fairness metrics:** Disparate Impact, Equal Opportunity Difference
- Visualizes the trade-off between model performance and fairness
- Provides comprehensive result analysis

---

## Dataset
This project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**:

- 24 professional actors (12 female, 12 male)  
- Speaking and singing with various emotions  
- 8 emotion categories: neutral, calm, happy, sad, angry, fearful, disgust, surprised

### Classification Focus
- **Label 1:** Neutral (class `0`)  
- **Label 3:** Happy (class `1`)  
- **Protected Attribute:** Gender (`female: 0`, `male: 1`)

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Required packages (install with `pip install -r requirements.txt`):
  - numpy  
  - pandas  
  - librosa  
  - matplotlib  
  - scikit-learn  
  - aif360

### Setup
1. Clone this repository or download the source code  
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the RAVDESS dataset from [Zenodo](https://zenodo.org/record/1188976)  
4. Extract the audio files to the `Audio_Speech_Actors_01-24` directory  
5. Run the experiment:
   ```bash
   python fair_speech_recognition.py
   ```

---

## Usage

The script supports the following command-line arguments:

```bash
python fair_speech_recognition.py --data_dir PATH_TO_DATA --output_dir results --bias_drop 0.4 --eta 25.0
```

### Available Arguments

| Argument         | Description                                                     | Default                     |
|------------------|-----------------------------------------------------------------|-----------------------------|
| `--data_dir`     | Directory containing audio files                                | `Audio_Speech_Actors_01-24` |
| `--output_dir`   | Directory to save results and plots                             | `results`                   |
| `--sample_rate`  | Sample rate for audio processing                                | `22050`                     |
| `--n_mfcc`       | Number of MFCC features to extract                              | `13`                        |
| `--bias_drop`    | Fraction of male-happy samples to drop for bias induction       | `0.4`                       |
| `--eta`          | Eta parameter for Prejudice Remover                             | `25.0`                       |
| `--random_state` | Random seed for reproducibility                                 | `42`                        |
| `--n_splits`     | Number of cross-validation splits                               | `5`                         |
| `--no_plots`     | Disable plot generation (set this flag to skip visualizations)  | *(not set)*                 |

---
