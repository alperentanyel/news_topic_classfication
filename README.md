# Multi-Category TF-IDF GloVe Text Classifier

A sophisticated text classification system that combines multi-category TF-IDF weighting with GloVe word embeddings for news article categorization. This system achieves **89.97% accuracy** on a 4-category news classification task using advanced cross-category weighting and intelligent missing word handling.

## 🎯 Project Overview

This project implements an advanced text classification pipeline specifically designed for categorizing news articles into four main categories:
- **World** (Politics/International News) - Category 1
- **Sports** (Sports News) - Category 2  
- **Business** (Business/Economic News) - Category 3
- **Sci/Tech** (Science/Technology News) - Category 4

## 🏗️ Architecture & Key Features

### Core Components

1. **Multi-Category TF-IDF Vectorization**
   - Separate TF-IDF vectorizers for each category (4 vectorizers total)
   - Cross-category weight calculation for enhanced discrimination
   - Maximum 5,000 features per category for balanced representation

2. **GloVe Word Embeddings Integration**
   - 300-dimensional GloVe vectors (glove.6B.300d)
   - Contextual word representation for semantic understanding
   - Smart missing word handling with semantic similarity

3. **Advanced Missing Word Management**
   - Semantic similarity-based word replacement using cosine similarity
   - Context-aware vector generation for OOV (Out-of-Vocabulary) words
   - KNN-like approach for finding optimal word replacements
   - Caching system for efficient processing

4. **Dimensionality Reduction**
   - PCA reduction from 1200D to 800D
   - Preserves essential information while reducing computational complexity
   - Explained variance ratio tracking

### Technical Innovations

#### Cross-Category Weighting System
The system calculates discriminative weights by comparing a word's TF-IDF score in the target category against its scores in other categories:

```python
# Statistical normalization with z-score based distinctiveness
z_score = (target_score - mean_other) / std_other
normalized_distinctiveness = max(0, min(2, z_score)) / 2.0
final_weight = target_score * (1.0 + normalized_distinctiveness)
```

#### Intelligent Missing Word Handling
- **Edit Distance Calculation**: Levenshtein distance for word similarity
- **Semantic Vector Creation**: Weighted average of similar words using cosine similarity
- **Contextual Analysis**: Window-based context consideration (±2 words)
- **Category-Aware Scoring**: Category-specific weight assignment

## 📊 Performance Metrics

- **Best Accuracy**: 89.97%
- **Vector Dimensions**: 1200D → 800D (PCA)
- **Categories**: 4 (World, Sports, Business, Sci/Tech)
- **Cross-validation**: 5-fold CV for robust evaluation

### Model Comparison Results
```
Model                CV Score      Test Acc     Time(s)   Dimension
LR_PCA              0.8924±0.003   0.8997       45.2s     800D
LR_NO_PCA           0.8891±0.004   0.8945       78.6s     1200D
SVM_PCA             0.8856±0.005   0.8923       156.7s    800D
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn tqdm requests
```

### Required Libraries
```python
import pandas as pd
import numpy as np
import pickle
import re
import os
import time
import warnings
import requests
import zipfile
import io
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import _stop_words
```

### Data Structure
```
archive/
├── train.csv    # Training data (label, title, description)
└── test.csv     # Test data (label, title, description)
```

**CSV Format:**
- Column 1: Category label (1-4)
- Column 2: Article title
- Column 3: Article description

### Running the Classification
```bash
python ml_multi_category_tfidf_faster.py
```

The system will automatically:
1. Download GloVe embeddings (if not present)
2. Analyze missing words in the dataset
3. Train multi-category TF-IDF vectorizers
4. Build word score cache for optimization
5. Train and evaluate multiple models
6. Save the best performing model

## 🔧 Configuration

Key parameters in the configuration:
```python
# Core Settings
GLOVE_DIM = 300                    # GloVe vector dimension
TFIDF_MAX_FEATURES = 5000         # Features per category
FINAL_DIM = 800                   # PCA output dimension
cv = 5                            # Cross-validation folds

# Data Paths
train_path = "archive/train.csv"
test_path = "archive/test.csv"
models_dir = "models"
glove_dir = "glove"

# Optional Settings
SAMPLE_SIZE = None                # For quick testing (set to int for sampling)
MAX_WORDS = 15000                # Maximum vocabulary size
```

## 📁 File Structure

```
topic_classifier/
├── ml_multi_category_tfidf_faster.py    # Main classification script
├── README.md                            # This file
├── models/                              # Saved models directory
│   ├── fast_contextual_logistic_regression.pkl
│   ├── fast_contextual_vectorizer.pkl
│   └── ...
├── glove/                              # GloVe embeddings
│   ├── glove.6B.50d.txt
│   ├── glove.6B.100d.txt
│   ├── glove.6B.200d.txt
│   └── glove.6B.300d.txt
└── archive/                            # Dataset
    ├── train.csv
    ├── test.csv
    └── ...
```

## 🎯 Key Algorithms

### 1. Multi-Category TF-IDF Vectorization
Each category has its own TF-IDF vectorizer trained on category-specific text data:
- Separate feature spaces for better category discrimination
- Cross-category weight calculation for enhanced performance
- Stop word removal and text preprocessing

### 2. Cross-Category Weight Calculation
```python
def calculate_cross_category_weight(token, target_category, category_scores):
    """
    Calculate discriminative weight based on target category vs other categories
    """
    target_score = category_scores[target_category]
    other_scores = [score for cat, score in category_scores.items() if cat != target_category]
    avg_other_score = np.mean(other_scores)
    
    # Distinctiveness: high in target, low in others
    distinctiveness = target_score / (avg_other_score + 0.01)
    final_weight = target_score * min(max(distinctiveness, 0.1), 3.0)
    
    return min(final_weight, 2.0)
```

### 3. Missing Word Vector Generation
For words not found in GloVe embeddings:
```python
def create_semantic_missing_word_vector(word, embeddings_index, context_words, target_category):
    """
    Generate vector for missing word using semantic similarity and context
    """
    # 1. Find semantically similar words using cosine similarity
    similar_words = find_similar_words_semantic(word, embeddings_index, context_words)
    
    # 2. Calculate context-aware weights
    weights = calculate_contextual_weights(similar_words, context_words, target_category)
    
    # 3. Generate weighted average vector
    vectors = [embeddings_index[word] for word in similar_words]
    return np.average(vectors, axis=0, weights=weights)
```

## 📈 Future Improvements

### Enhanced Missing Word Processing
The current system uses semantic similarity and edit distance for handling out-of-vocabulary words. **Planned improvements** include:

#### 1. Neighborhood-Based Scoring
- **Implementation**: Analyze k-nearest neighbors in the embedding space for missing words
- **Weight Assignment**: Calculate category-specific scores based on neighboring words' category performance
- **Dynamic Adjustment**: Real-time weight updates based on local context density

```python
# Future enhancement concept
def enhanced_missing_word_vector(word, context, category, k=10):
    """
    Generate missing word vector using k-nearest neighbors
    with category-specific scoring
    """
    # Find k-nearest semantic neighbors
    neighbors = find_k_nearest_semantic_words(word, embeddings_index, k=k)
    
    # Calculate category scores for each neighbor
    category_scores = []
    for neighbor in neighbors:
        if neighbor in word_score_cache:
            score = word_score_cache[neighbor].get(category, 0.0)
            category_scores.append(score)
        else:
            category_scores.append(0.1)  # Default score
    
    # Weight neighbors by category relevance and semantic similarity
    semantic_weights = calculate_semantic_similarities(word, neighbors)
    category_weights = normalize_weights(category_scores)
    final_weights = combine_weights(semantic_weights, category_weights)
    
    # Generate weighted vector
    neighbor_vectors = [embeddings_index[neighbor] for neighbor in neighbors]
    return weighted_average(neighbor_vectors, final_weights)
```

#### 2. Advanced Contextual Embeddings
- **Transformer Integration**: Incorporate BERT/RoBERTa for context-dependent representations
- **Fine-tuning**: Domain-specific fine-tuning on news data
- **Dynamic Contextualization**: Real-time context adaptation

#### 3. Adaptive Category Weighting
- **Dynamic Weights**: Self-adjusting category weights based on classification feedback
- **Category Evolution**: Learning from misclassifications to improve category boundaries
- **Real-time Optimization**: Continuous improvement during inference

### Implementation Roadmap

1. **Phase 1**: K-nearest neighbor enhancement for missing words
2. **Phase 2**: Integration with transformer-based embeddings
3. **Phase 3**: Dynamic weight adaptation system
4. **Phase 4**: Real-time learning capabilities

## 🔬 Technical Details

### Text Preprocessing Pipeline
```python
def enhanced_clean_text(text):
    """Advanced text preprocessing with multiple cleaning steps"""
    # 1. Lowercase conversion
    # 2. URL and email removal
    # 3. Number removal
    # 4. Punctuation cleaning
    # 5. Short/long word filtering
    # 6. Repeated character normalization
    # 7. Whitespace normalization
```

### Missing Word Analysis
The system provides comprehensive analysis of out-of-vocabulary words:
- **Missing word ratio**: Typically 8-12% of vocabulary
- **Categories**: Numerical, technical terms, proper nouns, spelling errors
- **Handling strategy**: Semantic replacement with contextual weighting

Sample missing word analysis output:
```
📊 MISSING WORD ANALYSIS:
   Total words: 45,231
   Missing words (unique): 3,847
   Missing words (total): 5,124
   Missing word ratio: 11.33%

🔍 MISSING WORD CATEGORIES:
   Technical Terms: 892 words
   Proper Nouns: 654 words
   Spelling Errors: 432 words
   Very Long (15+ chars): 287 words
   Mixed (letter+number): 156 words
```

### Performance Optimization
- **Caching System**: Global word score cache for repeated calculations
- **Batch Processing**: Memory-efficient processing for large datasets
- **Vectorized Operations**: NumPy optimizations for matrix operations
- **Sparse Matrix Support**: Efficient handling of TF-IDF sparse matrices

## 📋 Usage Examples

### Basic Usage
```python
from ml_multi_category_tfidf_faster import MultiCategoryTFIDFGloVeVectorizer

# Load your data
train_texts = ["Sample news article text...", ...]
train_labels = [1, 2, 3, 4, ...]  # Categories 1-4

# Initialize vectorizer
vectorizer = MultiCategoryTFIDFGloVeVectorizer(
    embeddings_index=embeddings_index,
    dim=300,
    max_features_per_category=5000,
    final_dim=800,
    handle_missing_words=True
)

# Train and transform
X_train = vectorizer.fit_transform(train_texts, train_labels)

# Train classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=2500)
classifier.fit(X_train, train_labels)

# Predict new texts
X_test = vectorizer.transform(test_texts)
predictions = classifier.predict(X_test)
```

### Advanced Configuration
```python
# Custom configuration for specific use cases
vectorizer = MultiCategoryTFIDFGloVeVectorizer(
    embeddings_index=embeddings_index,
    dim=300,                           # GloVe dimension
    max_features_per_category=7500,    # More features for complex domains
    final_dim=1000,                    # Higher dimensional output
    handle_missing_words=True          # Enable smart missing word handling
)
```

## 🏆 Results & Evaluation

### Classification Report
```
              precision    recall  f1-score   support

       World       0.91      0.89      0.90      1900
      Sports       0.89      0.92      0.90      1900
    Business       0.88      0.87      0.87      1900
    Sci/Tech       0.91      0.92      0.91      1900

    accuracy                           0.90      7600
   macro avg       0.90      0.90      0.90      7600
weighted avg       0.90      0.90      0.90      7600
```

### Cross-Validation Results
- **Mean Accuracy**: 89.24% ± 0.3%
- **Consistency**: Low standard deviation indicates stable performance
- **Robustness**: Performs well across different data splits

## 🤝 Contributing

This project is part of an academic research initiative. Contributions are welcome for:
- Algorithm improvements
- Missing word handling enhancements
- Performance optimizations
- Documentation improvements

## 📚 References

1. **GloVe**: Pennington, J., Socher, R., & Manning, C. (2014). Glove: Global vectors for word representation.
2. **TF-IDF**: Sparck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval.
3. **Cross-Category Analysis**: Custom methodology developed for multi-category text classification.

## 📝 License

This project is developed for academic purposes as part of YAP470 coursework.

---

**Note**: This implementation represents a state-of-the-art approach to text classification, combining traditional TF-IDF methods with modern word embeddings and intelligent missing word handling for superior performance on news categorization tasks.
