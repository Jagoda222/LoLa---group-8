# **Curriculum Learning with Complexity Measures**

## **Project Overview**
This project explores how **curriculum learning**—ordering training samples by complexity—can impact the performance of fine-tuned language models. The notebook calculates several complexity measures on textual data and evaluates different training strategies, including **increasing, decreasing, and baseline orderings** of the samples.

The main goal is to understand whether presenting samples in a specific order improves model learning efficiency and performance.

---

## **Contents**
- **Notebook:** `calculate_measure.ipynb` — Contains all steps from data preparation to model fine-tuning and evaluation.
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

### 1. **Imports and Setup:**
   Install and import the necessary libraries.

### 2. **Dataset Loading and Preprocessing:**
   - **Initial Random Sampling:** First, 30,000 random rows were extracted from the SNLI dataset while ensuring that the dataset maintained the triple structure (premise, hypothesis, and label).
   - **Final Sampling for Complexity Distribution:** Using a custom function that calculates complexity measures, 2,100 rows were sampled from the initial subset while preserving the distribution of the calculated complexity values.

### 3. **Calculating Complexity Measures:**
   The notebook calculates several complexity measures, each designed to evaluate different aspects of sentence difficulty:

   - **Syntactic Overlap + Lexical Diversity + Sentence Length (SO + LD + SL):** Combines syntactic similarity between premise and hypothesis, lexical diversity of words, and sentence length.
   - **Syntactic Tree Depth Measure:** Calculates the depth of the syntactic parse tree for a sentence to assess its structural complexity.
   - **Sentence Length Measure:** The number of words in the sentence, representing basic length-based complexity.
   - **Flesch-Kincaid Measure:** A readability metric based on word length and sentence structure, commonly used to assess how difficult a text is to read.
   - **Sentence Length and Flesch-Kincaid Combined Measure:** Combines the sentence length and Flesch-Kincaid scores to provide a holistic view of both syntactic and semantic complexity.

   The calculated measures are saved to CSV files for later use during training.

### 4. **Ordering the Data:**
   Generate versions of the dataset ordered by:
   - **Increasing complexity:** From simpler to more complex examples.
   - **Decreasing complexity:** From more complex to simpler examples.
   - **Baseline (random order):** Random shuffling of the data.

### 5. **Fine-tuning Pre-trained Models:**
   Use models such as GPT-2 and DeBERTa to fine-tune on the ordered datasets.

### 6. **Evaluation and Results:**
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

