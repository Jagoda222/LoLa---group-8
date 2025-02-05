# **Curriculum Learning with Complexity Measures**

## **Project Overview**
This project explores how **curriculum learning**—ordering training samples by complexity—can impact the performance of fine-tuned language models. The notebook calculates several complexity measures on textual data and evaluates different training strategies, including **increasing, decreasing, and baseline orderings** of the samples.

The main goal is to understand whether presenting samples in a specific order improves model learning efficiency and performance.

---

## **Contents**
- **Notebook:** `curriculum_learning.ipynb` — Contains all steps from data preparation to model fine-tuning and evaluation.
- **Dependencies:** Required libraries and installation instructions.

---

## **Dependencies**
Make sure you have the following libraries installed:

- `datasets`
- `transformers`
- `spacy`
- `evaluate`
- `nltk`
- `syllapy`
- `torch`
- `pandas`

You can install them using:

```bash
pip install -U datasets transformers evaluate spacy nltk syllapy torch pandas accelerate
python -m spacy download en_core_web_md
```

---

## **How to Run the Notebook**
1. **Download the project or open the notebook** in a Colab environment.
2. **Install dependencies** using the instructions above.
3. **Run the cells sequentially** in the notebook.

---

## **Notebook Overview**
The notebook is divided into several key sections:

1. **Imports and Setup:**  
   Install and import the necessary libraries.

2. **Dataset Loading and Preprocessing:**  
   - Load the SNLI dataset or other datasets of interest.
   - Apply basic preprocessing to prepare for complexity measure calculation.

3. **Calculating Complexity Measures:**  
   - Calculate various metrics, including:
     - Sentence length
     - Syntactic tree depth
     - Readability (Flesch-Kincaid scores)
   - Save the calculated measures to CSV files for training.

4. **Ordering the Data:**  
   Generate versions of the dataset ordered by:
   - **Increasing complexity**
   - **Decreasing complexity**
   - **Baseline (random order)**

5. **Fine-tuning Pre-trained Models:**  
   Use models such as GPT-2 and DeBERTa to fine-tune on the ordered datasets.

6. **Evaluation and Results:**  
   Evaluate model performance across the different orderings using accuracy or other metrics.

---

## **Project Results**
The project aims to identify:
- Whether ordering the data by complexity improves model training.
- How the performance differs between increasing, decreasing, and baseline orderings.

---

## **Possible Future Work**
- Add more complexity measures, such as semantic complexity.
- Extend the evaluation to different datasets beyond SNLI.
- Explore curriculum learning strategies using dynamic ordering during training.

