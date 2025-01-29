"""
!pip install datasets==2.14.5
!pip install spacy
!pip install syllapy
!python -m spacy download en_core_web_sm
"""

import spacy
import pickle
import nltk
import syllapy
from datasets import load_dataset, load_metric, concatenate_datasets

### PASTE DATA FILE HERE
snli_data = load_dataset("snli")["train"].select(range(2000))


#syntactic tree depth metric -----------------------------------------------------------------------------
def tree_depth_metric(data): ### function runs on training data (so not on )
    
    nlp = spacy.load("en_core_web_sm")
    premise_docs = list(nlp.pipe(data["premise"]))
    hypothesis_docs = list(nlp.pipe(data["hypothesis"]))
    premise_tree_depth = []
    hypothesis_tree_depth = []

    def get_dependency_depth(token):
        """Recursively find the depth of a token in the dependency tree."""
        if token == token.head:  # Root node (self-referential head)
            return 0
        return 1 + get_dependency_depth(token.head)
    
    for doc in premise_docs:
        max_depth = max(get_dependency_depth(token) for token in doc)
        premise_tree_depth.append(max_depth)

    for doc in hypothesis_docs:
        max_depth = max(get_dependency_depth(token) for token in doc)
        hypothesis_tree_depth.append(max_depth)

    data = data.add_column("premise_tree_depth", premise_tree_depth)
    data = data.add_column("hypothesis_tree_depth", hypothesis_tree_depth)

    return data

snli_data = tree_depth_metric(snli_data)
combined_scores = [
    snli_data["premise_tree_depth"][i] + snli_data["hypothesis_tree_depth"][i] 
    for i in range(len(snli_data["premise"]))
]
snli_data = snli_data.add_column("combined_depth", combined_scores)

print("Tree depth metric calculated....")

# Flesch kincaid grade level metric -----------------------------------------------------------------------------
def flesch_kincaid(data):

  def flesch_kincaid_calc(text):
      # Tokenize text
      sentences = nltk.sent_tokenize(text)
      words = nltk.word_tokenize(text)

      # Count total words and sentences
      total_words = len(words)
      total_sentences = len(sentences)

      # Count total syllables in the text
      total_syllables = sum(syllapy.count(word) for word in words)

      # Flesch-Kincaid Grade Level formula
      grade_level = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
      return grade_level

  premise_metric = []
  hypothesis_metric = []
  both_metric = []

  for text in data["premise"]:
    premise_metric.append(flesch_kincaid_calc(text))

  for text in data["hypothesis"]:
    hypothesis_metric.append(flesch_kincaid_calc(text))

  for i in range(len(premise_metric)):
    both_metric.append(premise_metric[i] + hypothesis_metric[i])

  data = data.add_column("premise_metric", premise_metric)
  data = data.add_column("hypothesis_metric", hypothesis_metric)
  data = data.add_column("both_metric", both_metric)

  return data
  

snli_data = flesch_kincaid(snli_data)

print("Flesch-kincaid metric calculated....")

# Semantic similarity metric -----------------------------------------------------------------------------

def semantic_similarity_metric(data):
    
    nlp = spacy.load("en_core_web_sm")
    premise_docs = list(nlp.pipe(data["premise"]))
    hypothesis_docs = list(nlp.pipe(data["hypothesis"]))
    
    similarity_scores = []

    for i in range(len(premise_docs)):
        similarity_score = premise_docs[i].similarity(hypothesis_docs[i])
        similarity_scores.append(similarity_score)

    data = data.add_column("semantic_similarity", similarity_scores)

    return data

snli_data = semantic_similarity_metric(snli_data)

print("Semantic similarity metric calculated....")

# Semantic similarity metric -----------------------------------------------------------------------------
def jaccard_coefficient(data):

  coefficient_scores = []

  def jaccard_coefficient_calc(sentence1, sentence2):
      # Tokenize the sentences into words
      set1 = set(sentence1.lower().split())
      set2 = set(sentence2.lower().split())
      
      # Calculate intersection and union
      intersection = set1.intersection(set2)
      union = set1.union(set2)
      
      # Calculate Jaccard Coefficient
      jaccard = len(intersection) / len(union)
      return jaccard
  
  for i in range(len(data["premise"])):
    coefficient_scores.append(jaccard_coefficient_calc(data["premise"][i], data["hypothesis"][i]))

  data = data.add_column("jaccard_coeff", coefficient_scores)

  return data

snli_data = jaccard_coefficient(snli_data)

print("Jaccard coefficient metric calculated....")

# Dumping the data in a .pkl file -----------------------------------------------------------------------------
with open("data_with_metric.pkl", "wb") as file:  # Open the file in write-binary mode
    pickle.dump(snli_data, file)