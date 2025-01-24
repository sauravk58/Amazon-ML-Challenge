# Machine Learning Pipeline for Entity-Related Measurements Extraction [ Amazon ML Challenge ]

## Introduction

This project focuses on developing a machine learning pipeline that processes images to extract specific entity-related measurements, such as width, height, depth, item weight, and item volume, using Optical Character Recognition. The core task is to extract numerical values and measurement units from product images and map them to relevant entities, such as product dimensions or weight.

## ML Models Used:
1. **EasyOCR**
2. **Pattern Matching and Regular Expressions**  
   [While not a traditional machine learning model, regular expressions (regex) were used.]
3. **ResNet-50**
4. **BERT**

## Machine Learning Approach:

### 1. Data Preprocessing
The dataset contains image URLs and corresponding entity names and values. Preprocessing involves:
- **OCR**: Tesseract is used to extract textual content from the images.
- **Text Post-processing**: The extracted text is cleaned by removing unwanted characters (non-alphanumeric) and standardizing units (e.g., replacing `"` with `inch`).

### 2. Feature Extraction
A combination of CNNs for image feature extraction and BERT for text processing is used.

### 3. Unit and Value Extraction
After extracting the text from an image, the goal is to identify numbers and their associated units. A predefined mapping between entities and allowed units is utilized.

- **Entity-Unit Mapping**: A dictionary (`entity_unit_map`) is created to map each entity (e.g., "width," "height") to the set of allowed measurement units (e.g., centimetre, inch, metre).
- **Pattern Matching for Units**: A regular expression is used to detect numerical values followed by units in the extracted text. The regex is applied for each unit type, searching for values like "15 cm" or "3.5 kg."

### 4. Value and Unit Selection
- **Relevance Filtering**: The program ensures that only units relevant to the entity are selected. If multiple values are found, the most appropriate one is chosen based on the entity type.

### 5. Predictions
The final step is storing the extracted values for each image and entity in a predictions file. The predictions are saved as a CSV file.

## Experiments-1:

### 1. Hybrid Model Architecture
- **ResNet-50**: A pre-trained ResNet-50 model is used to extract high-level features from images. Its final layer is replaced with a smaller fully connected layer (128 units) to suit the task.
- **BERT**: A pre-trained BERT model is used to process text extracted by the OCR engine. The `[CLS]` token embedding is used as the text feature representation.
- **Embeddings for Metadata**

### 2. Loss Functions
- **Value Prediction**: Mean squared error (MSE) loss is used for predicting the entity value (e.g., the number before a unit like "12 cm").
- **Unit Prediction**: Cross-entropy loss is used for predicting the correct unit (e.g., "cm").

### 3. Hyperparameter Tuning
To fine-tune hyperparameters such as learning rate and batch size, we used **Optuna**, an automated hyperparameter optimization framework. The objective function was minimized by experimenting with different combinations of hyperparameters.

## Experiments-2:

### 1. Data Preprocessing:
Values are extracted from raw text using regex patterns. These are split into numeric values (e.g., "12") and units (e.g., "cm").

### 2. Text Extraction Using OCR
Tesseract is used to extract textual content from the images.

### 3. Unit and Value Extraction

### 4. Value and Unit Selection

### 5. Predictions
This code is present in `app.ipynb` file.

## Conclusion:
This OCR-based approach effectively extracts entity-specific values from product images, enabling automated data processing. The pipeline, using EasyOCR and entity-specific unit filtering, is scalable across large datasets. However, with an F1 score of 0.44, there's room for improvement. Enhancing text post-processing and error handling could boost robustness, especially for low-quality images or complex layouts. The combination of OCR and regex is reliable, but refining these methods could further improve accuracy and performance.

## Final Accuracy and Ranking:


![amazon](https://github.com/user-attachments/assets/ce569b29-6c4e-45d6-8801-85d229a108fe)

