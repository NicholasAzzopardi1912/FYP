# Integrating Privileged Information in Different Affect Modelling Paradigms

**Nicholas Azzopardi**
University of Malta — Faculty of ICT, Department of Artificial Intelligence

---

## Repository Structure
FYP/
├── Recola_Preprocessing.py
├── split_modalities.py
│
├── Classification Baseline Models/
│ ├── Classification_single_mod_audio.py
│ ├── Classification_single_mod_physio.py
│ └── Classification_single_mod_video.py
│
├── Regression Baseline Models/
│ ├── Regression_single_mod_audio.py
│ ├── Regression_single_mod_physio.py
│ └── Regression_single_mod_video.py
│
├── Teacher Models/
│ ├── Classification_3stream_Teacher.py
│ └── Regression_3stream_Teacher.py
│
├── Student Models/
│ ├── Classification Student Models/
│ │ ├── 1 Cosine Distance on Representations/
│ │ │ ├── AudioClassificationStudent_CosineDistance.py
│ │ │ ├── PhysioClassificationStudent_CosineDistance.py
│ │ │ └── VideoClassificationStudent_CosineDistance.py
│ │ ├── 2 KL Divergence on Probabilistic outputs/
│ │ │ ├── AudioClassificationStudent_KL_Divergence.py
│ │ │ ├── PhysioClassificationStudent_KL_Divergence.py
│ │ │ └── VideoClassificationStudent_KL_Divergence.py
│ │ └── 3 Combined Cosine + KL/
│ │ ├── AudioClassificationStudent_Combined_CosineDist+KL.py
│ │ ├── PhysioClassificationStudent_Combined_CosineDist+KL.py
│ │ └── VideoClassificationStudent_Combined_CosineDist+KL.py
│ └── Regression Student Models/
│ ├── AudioRegressionStudent.py
│ ├── PhysioRegressionStudent.py
│ └── VideoRegressionStudent.py
│
└── Results/
├── Single Mod Class Results/
│ ├── Audio/
│ ├── Physio/
│ └── Video/
├── Single Mod Reg Results/
│ ├── Audio Results/
│ ├── Physio Results/
│ └── Video Results/
├── Teacher Classification Results/
├── Teacher Regression Results/
├── Student Classification Results/
│ ├── 1 Representations using Cosine Distance/
│ ├── 2 Predicted Probabilities using KL Divergence/
│ └── 3 Combined Loss Terms Cosine Distance and KL/
└── Student Regression Results/
├── Audio Student Regression Model/
├── Physio Student Regression Model/
└── Video Student Regression Model/

---

## Directory Overview

### Root
- **`Recola_Preprocessing.py`** — Preprocesses the raw RECOLA dataset by standardising all input features to zero mean and unit variance using StandardScaler, and generates binary classification targets for arousal and valence via a global median-split threshold. Outputs a processed dataset containing standardised features, all four target columns, and participant identifiers.
- **`split_modalities.py`** — Partitions the processed dataset into three modality-specific subsets (audio, video, physiological) based on RECOLA feature naming conventions. Each subset retains the participant identifier and all target columns, and is saved as a separate file to serve as input to the corresponding unimodal models.

### Classification Baseline Models
Single-modality feedforward neural networks for binary classification of arousal and valence. Each model consists of three hidden layers (128, 64, 32 units) with ReLU activations and dropout regularisation, a sigmoid output layer with binary cross-entropy loss, and is trained with the Adam optimiser under an 18-fold leave-one-participant-out (LOPO) cross-validation protocol. These models serve as reference performance in the absence of privileged information.

### Regression Baseline Models
Single-modality feedforward neural networks for continuous arousal and valence prediction, sharing the same architecture as the classification baselines but using a linear output layer with MSE loss. These establish the unimodal performance reference for the regression paradigm prior to the introduction of privileged information.

### Teacher Models
- **`Classification_3stream_Teacher.py`** — A standalone three-stream neural network that fuses audio, video, and physiological modalities through dedicated encoder branches (128 and 64 units, ReLU, L2 regularisation, dropout 0.3). The three branch outputs are concatenated into a 192-dimensional embedding and passed through a fusion head (128 and 32 units) before a sigmoid output layer. Trained independently under the same LOPO protocol, this script serves purely as the multimodal upper-bound performance reference for the classification paradigm. The identical teacher architecture is re-implemented and retrained within each classification student script as part of the sequential teacher-student training per fold.
- **`Regression_3stream_Teacher.py`** — Identical three-stream architecture with a linear output and MSE loss. Serves as the standalone upper-bound performance reference for the regression paradigm. As with classification, the same teacher architecture is re-implemented and retrained within each regression student script during the sequential per-fold training procedure.

### Student Models — Classification
Single-modality student models trained under the LUPI framework, guided by the frozen tri-modal classification teacher. All three setups share the same architecture as the classification baseline, with the 32-dimensional third hidden layer acting as the student's bottleneck representation. The teacher's influence is controlled by the weighting parameter α ∈ {0.25, 0.50, 0.75, 1.0}, evaluated across all configurations:

- **1 Cosine Distance on Representations** — Feature-based distillation that aligns the student's bottleneck representation with the teacher's frozen 32-dimensional fusion layer output using squared cosine distance as the privileged loss term.
- **2 KL Divergence on Probabilistic Outputs** — Response-based distillation that transfers knowledge at the output level by minimising the Kullback-Leibler divergence between the teacher's and student's sigmoid output distributions.
- **3 Combined Cosine + KL** — A hybrid distillation strategy that combines both representation alignment and prediction distillation simultaneously as a joint privileged loss term, providing complementary feature-based and response-based knowledge transfer.

### Student Models — Regression
Single-modality student regression models trained under the LUPI framework using a single distillation strategy: representation alignment via squared cosine distance between the teacher's frozen 32-dimensional fusion layer and the student's bottleneck layer. The same α values are evaluated as in the classification students.

### Results
All experiment outputs are saved as CSV files per model, target variable (arousal/valence), and α configuration, under the 18-fold LOPO cross-validation protocol. Subfolders follow a consistent Audio / Physio / Video split across modalities:
- **Single Mod Class / Reg Results** — Per-fold metric outputs from the single-modality baseline models for classification and regression respectively.
- **Teacher Classification / Regression Results** — Per-fold metric outputs from the tri-modal teacher models, representing the multimodal upper-bound performance reference.
- **Student Classification Results** — Per-strategy distillation results across all three modalities, affective dimensions, and α values for the classification paradigm.
- **Student Regression Results** — Per-fold metric outputs from the regression student models across all three modalities, affective dimensions, and α values.