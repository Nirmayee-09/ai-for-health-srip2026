<<<<<<< HEAD
# AI for Health — SRIP 2026
Detecting breathing irregularities during sleep using physiological time-series data.

## Project Structure
```
internship/
├── Data/              → raw participant data (AP01–AP05)
├── Dataset/           → processed dataset (breathing_dataset.csv)
├── Visualizations/    → per-participant PDF plots
├── Results/           → confusion matrix image
├── scripts/
│   ├── utils.py       → signal reading utilities
│   ├── vis.py         → visualization script
│   ├── create_dataset.py → preprocessing + dataset creation
│   └── train_model.py → 1D CNN training + evaluation
├── README.md
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Visualize signals
```bash
python scripts/vis.py -name "Data/AP01"
```
Generates a multi-page PDF in `Visualizations/` with nasal flow, thoracic movement, and SpO2 plotted with breathing events overlaid.

### Step 2: Create dataset
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
Filters signals to breathing frequency range (0.17–0.4 Hz), splits into 30s windows with 50% overlap, labels each window, saves as CSV.

### Step 3: Train and evaluate model
```bash
python scripts/train_model.py
```
Trains a 1D CNN using Leave-One-Participant-Out cross validation with SMOTE for class balancing. Reports accuracy, precision, recall, and confusion matrix.

## Results
| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.838 |
| Precision | 0.837 |
| Recall    | 0.838 |

## Notes
- GPU warnings from TensorFlow can be ignored if running on CPU
- Body event and Mixed Apnea classes removed due to insufficient samples (< 6)
- AI tools used: Claude (acknowledged as per assignment instructions)


### requirements.txt
pandas
numpy
scipy
matplotlib
tensorflow==2.10.0
scikit-learn
imbalanced-learn

=======
# AI for Health — SRIP 2026
Detecting breathing irregularities during sleep using physiological time-series data.

## Project Structure
```
internship/
├── Data/              → raw participant data (AP01–AP05)
├── Dataset/           → processed dataset (breathing_dataset.csv)
├── Visualizations/    → per-participant PDF plots
├── Results/           → confusion matrix image
├── scripts/
│   ├── utils.py       → signal reading utilities
│   ├── vis.py         → visualization script
│   ├── create_dataset.py → preprocessing + dataset creation
│   └── train_model.py → 1D CNN training + evaluation
├── README.md
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Visualize signals
```bash
python scripts/vis.py -name "Data/AP01"
```
Generates a multi-page PDF in `Visualizations/` with nasal flow, thoracic movement, and SpO2 plotted with breathing events overlaid.

### Step 2: Create dataset
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
Filters signals to breathing frequency range (0.17–0.4 Hz), splits into 30s windows with 50% overlap, labels each window, saves as CSV.

### Step 3: Train and evaluate model
```bash
python scripts/train_model.py
```
Trains a 1D CNN using Leave-One-Participant-Out cross validation with SMOTE for class balancing. Reports accuracy, precision, recall, and confusion matrix.

## Results
| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.838 |
| Precision | 0.837 |
| Recall    | 0.838 |

## Notes
- GPU warnings from TensorFlow can be ignored if running on CPU
- Body event and Mixed Apnea classes removed due to insufficient samples (< 6)
- AI tools used: Claude (acknowledged as per assignment instructions)


### requirements.txt
pandas
numpy
scipy
matplotlib
tensorflow==2.10.0
scikit-learn
imbalanced-learn

>>>>>>> f6d1247256e8435d41a89a736966ded0a1e8840f
