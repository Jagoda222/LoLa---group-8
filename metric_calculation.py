"""
To run this you need to use the following commands:
pip install datasets==2.14.5
pip install spacy
pip install syllapy
pip install nltk
pip install joblib
python -m spacy download en_core_web_sm
"""



""" 
This script calculates the metrics for a selected part of the dataset. It returns a .pkl file, 
which is in the same datasets.Dataset() format but with extra columns for the metrics.
"""
import spacy
import joblib
import nltk
import syllapy
from datasets import load_dataset
from spacy.tokens import DocBin

### PASTE DATA FILE HERE #####################################################
snli_data = load_dataset("snli")["train"]
print("Data loaded....")
### PASTE DATA FILE HERE #####################################################

#syntactic tree depth metric -----------------------------------------------------------------------------
def tree_depth_metric(data): 
    
    nlp = spacy.load("en_core_web_sm")
    premise_docs = list(DocBin().from_disk("docs_premises.spacy").get_docs(nlp.vocab))
    hypothesis_docs = list(DocBin().from_disk("docs_hypothesis.spacy").get_docs(nlp.vocab))

    
    premise_tree_depth = []
    hypothesis_tree_depth = []

    def get_dependency_depth(token):
        """Recursively find the depth of a token in the dependency tree."""
        if token == token.head:  # Root node (self-referential head)
            return 0
        return 1 + get_dependency_depth(token.head)
    
    for i, doc in enumerate(premise_docs):
        max_depth = max(get_dependency_depth(token) for token in doc)
        premise_tree_depth.append(max_depth)

        if i % 1000 == 0:
           print(i)

    for i, doc in enumerate(hypothesis_docs):
        max_depth = max(get_dependency_depth(token) for token in doc)
        hypothesis_tree_depth.append(max_depth)

        if i % 1000 == 0:
           print(i)

    premise_tree_depth = [premise_tree_depth[i//3] for i in range(3 * len(premise_tree_depth))]
    print("list thingy")
    data = data.add_column("premise_tree_depth", premise_tree_depth)
    print("premise column added")
    data = data.add_column("hypothesis_tree_depth", hypothesis_tree_depth)
    print("hypothesis column added")

    return data

def add_depths(data):
    data['combined_depth'] = data['premise_tree_depth'] + data['hypothesis_tree_depth']
    return data


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

  data = data.add_column("Flesch_kinaid_premise", premise_metric)
  data = data.add_column("Flesch_kinaid_hypothesis", hypothesis_metric)
  data = data.add_column("Flesch_kinaid_both", both_metric)

  return data
  

# Semantic similarity metric -----------------------------------------------------------------------------

def semantic_similarity_metric(data):
    
    nlp = spacy.load("en_core_web_md")
    premise_docs = list(DocBin().from_disk("docs_premises_md.spacy").get_docs(nlp.vocab))
    hypothesis_docs = list(DocBin().from_disk("docs_hypothesis.spacy").get_docs(nlp.vocab))
    premise_docs = [premise_docs[i//3] for i in range(3 * len(premise_docs))]
    similarity_scores = []

    for i in range(len(premise_docs)):
        similarity_score = premise_docs[i].similarity(hypothesis_docs[i])
        similarity_scores.append(similarity_score)

    data = data.add_column("semantic_similarity", similarity_scores)

    return data


# Sentence length metric ------------------------------------------------------------------------------------
def sentence_length(data):

  premise_length = []
  hypothesis_length = []
  combined_length = []

  def length_calc(text):
    return len(text.split())

  for text in data["premise"]:
    premise_length.append(length_calc(text))

  for text in data["hypothesis"]:
    hypothesis_length.append(length_calc(text))

  for i in range(len(premise_length)):
    combined_length.append(premise_length[i] + hypothesis_length[i])

  data = data.add_column("premise_length", premise_length)
  data = data.add_column("hypothesis_length", hypothesis_length)
  data = data.add_column("combined_length", combined_length)

  return data


# Dumping the data in a .pkl file -----------------------------------------------------------------------------
def run_calculation(data):

    # Syntactic tree depth
    snli_data = tree_depth_metric(data)
    combined_depth = snli_data.map(add_depths) # column added automatically
    print("combined thing calculated")

    print("Tree depth metric calculated....")
    
    # Flesch Kinaid 
    snli_data = flesch_kincaid(snli_data)
    print("Flesch-kincaid metric calculated....")

    # Semantic similarity
    snli_data = semantic_similarity_metric(snli_data)
    print("Semantic similarity metric calculated....")

    # Sentence length
    snli_data = sentence_length(snli_data)
    print("Sentence length metric calculated....")

    return snli_data
      

snli_data = run_calculation(snli_data)

with open("data_with_metrics.pkl", "wb") as file:  # Open the file in write-binary mode
    joblib.dump(snli_data, file)