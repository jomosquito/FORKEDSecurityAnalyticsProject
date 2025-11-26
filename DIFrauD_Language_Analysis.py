#!/usr/bin/env python3
"""
DIFrauD Multilingual Data Quality Assessment
Analyzing Language Diversity Impact on Classification Performance

Course: COSC 4371 Security Analytics - Fall 2025
Team Members: Joseph Mascardo, Niket Gupta

Project Objective:
Investigate whether all samples in the DIFrauD dataset are in English, analyze
language distribution by class and domain, and study the impact on classification performance.

Research Questions:
1. What is the language distribution across classes and domains in DIFrauD?
2. How does removing non-English samples affect classifier performance?
3. Do transformer-based models handle multilingual content better than traditional ML?

External Sources and References:
- DIFrauD Dataset: https://huggingface.co/datasets/difraud/difraud
- Citation: Boumber, D., et al. (2024). "Domain-Agnostic Adapter Architecture for Deception Detection." LREC-COLING 2024.
- langdetect: https://pypi.org/project/langdetect/
- datasets: https://huggingface.co/docs/datasets/
- transformers: https://huggingface.co/docs/transformers/
- scikit-learn: https://scikit-learn.org/
"""

# =============================================================================
# 1. Environment Setup and Imports
# =============================================================================

# Install required packages (run once):
# pip install datasets langdetect transformers torch scikit-learn pandas numpy matplotlib seaborn tqdm

import warnings
warnings.filterwarnings('ignore')

# Data handling
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Dataset loading - Source: https://huggingface.co/docs/datasets/
from datasets import load_dataset, concatenate_datasets

# Language detection - Source: https://pypi.org/project/langdetect/
from langdetect import detect, detect_langs, LangDetectException

# ML libraries - Source: https://scikit-learn.org/
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# Deep Learning - Source: https://huggingface.co/docs/transformers/
import torch
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress tracking
from tqdm import tqdm

# Statistical testing
from scipy import stats

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# =============================================================================
# 2. Load DIFrauD Dataset
# Dataset Source: https://huggingface.co/datasets/difraud/difraud
# =============================================================================

# Define all domains in DIFrauD dataset
DOMAINS = [
    'fake_news',
    'job_scams',
    'phishing',
    'political_statements',
    'product_reviews',
    'sms',
    'twitter_rumours'
]

def load_difraud_dataset():
    """
    Load all domains from DIFrauD dataset.
    Source: HuggingFace datasets library
    Dataset: https://huggingface.co/datasets/difraud/difraud
    """
    all_data = []

    for domain in tqdm(DOMAINS, desc="Loading domains"):
        try:
            # Load dataset for this domain
            dataset = load_dataset('difraud/difraud', domain)

            # Combine all splits
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    df_split = dataset[split].to_pandas()
                    df_split['domain'] = domain
                    df_split['split'] = split
                    all_data.append(df_split)

        except Exception as e:
            print(f"Error loading {domain}: {e}")

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    return df


# Load the dataset
print("Loading DIFrauD dataset from HuggingFace...")
df = load_difraud_dataset()

print(f"\nDataset loaded successfully!")
print(f"Total samples: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")


# Dataset overview
print("="*60)
print("DATASET OVERVIEW")
print("="*60)

print("\n--- Samples by Domain ---")
domain_counts = df.groupby('domain').agg({
    'text': 'count',
    'label': ['sum', 'mean']
}).round(3)
domain_counts.columns = ['Total', 'Deceptive', 'Deceptive_Ratio']
domain_counts['Non-Deceptive'] = domain_counts['Total'] - domain_counts['Deceptive']
print(domain_counts)

print("\n--- Overall Class Distribution ---")
print(f"Deceptive (label=1): {df['label'].sum():,} ({df['label'].mean()*100:.2f}%)")
print(f"Non-Deceptive (label=0): {(df['label']==0).sum():,} ({(1-df['label'].mean())*100:.2f}%)")

print("\n--- Sample Text Lengths ---")
df['text_length'] = df['text'].str.len()
print(df.groupby('domain')['text_length'].describe().round(1))


# =============================================================================
# 3. Language Detection Pipeline
# Source: langdetect library - https://pypi.org/project/langdetect/
# Note: langdetect is a port of Google's language-detection library
# =============================================================================

def detect_language_safe(text, min_length=20):
    """
    Safely detect language of text with error handling.

    Source: langdetect library (https://pypi.org/project/langdetect/)

    Parameters:
    - text: Input text string
    - min_length: Minimum text length for reliable detection

    Returns:
    - Tuple of (detected_language_code, confidence_score)
    """
    if not isinstance(text, str) or len(text.strip()) < min_length:
        return ('unknown', 0.0)

    try:
        # Get language probabilities
        langs = detect_langs(text)
        # Return top language and its probability
        top_lang = langs[0]
        return (top_lang.lang, top_lang.prob)
    except LangDetectException:
        return ('unknown', 0.0)
    except Exception as e:
        return ('error', 0.0)


# Test the function
test_texts = [
    "This is a test message in English.",
    "Ceci est un message de test en français.",
    "Pathaya enketa maraikara pa",  # From SMS dataset (Tamil)
    "短文本"  # Short Chinese text
]

print("Language Detection Test:")
for text in test_texts:
    lang, conf = detect_language_safe(text)
    print(f"  '{text[:40]}...' -> {lang} (conf: {conf:.2f})")


# Apply language detection to entire dataset
print("Detecting languages for all samples...")
print("(This may take several minutes)\n")

# Apply with progress bar
tqdm.pandas(desc="Detecting languages")
language_results = df['text'].progress_apply(detect_language_safe)

# Extract language codes and confidence scores
df['detected_language'] = language_results.apply(lambda x: x[0])
df['language_confidence'] = language_results.apply(lambda x: x[1])

print("\nLanguage detection completed!")
print(f"Unique languages detected: {df['detected_language'].nunique()}")


# =============================================================================
# 4. Language Distribution Analysis
# =============================================================================

# Overall language distribution
print("="*60)
print("OVERALL LANGUAGE DISTRIBUTION")
print("="*60)

lang_counts = df['detected_language'].value_counts()
lang_percentages = df['detected_language'].value_counts(normalize=True) * 100

lang_summary = pd.DataFrame({
    'Count': lang_counts,
    'Percentage': lang_percentages.round(2)
})
print(lang_summary.head(15))

# English vs Non-English
df['is_english'] = df['detected_language'] == 'en'
print(f"\n--- English vs Non-English ---")
print(f"English samples: {df['is_english'].sum():,} ({df['is_english'].mean()*100:.2f}%)")
print(f"Non-English samples: {(~df['is_english']).sum():,} ({(~df['is_english']).mean()*100:.2f}%)")


# Language distribution by CLASS (deceptive vs non-deceptive)
print("="*60)
print("LANGUAGE DISTRIBUTION BY CLASS")
print("="*60)

class_lang_dist = pd.crosstab(
    df['label'].map({0: 'Non-Deceptive', 1: 'Deceptive'}),
    df['is_english'].map({True: 'English', False: 'Non-English'}),
    margins=True
)
print("\nCounts:")
print(class_lang_dist)

# Percentages within each class
class_lang_pct = pd.crosstab(
    df['label'].map({0: 'Non-Deceptive', 1: 'Deceptive'}),
    df['is_english'].map({True: 'English', False: 'Non-English'}),
    normalize='index'
) * 100
print("\nPercentages (within each class):")
print(class_lang_pct.round(2))

# Chi-square test for class vs language
contingency = pd.crosstab(df['label'], df['is_english'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square test (Class vs Language):")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  p-value: {p_value:.4e}")
print(f"  Significant (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")


# Language distribution by DOMAIN
print("="*60)
print("LANGUAGE DISTRIBUTION BY DOMAIN")
print("="*60)

domain_lang_analysis = []

for domain in DOMAINS:
    domain_df = df[df['domain'] == domain]

    total = len(domain_df)
    english = domain_df['is_english'].sum()
    non_english = total - english

    # Top non-English languages
    non_eng_langs = domain_df[~domain_df['is_english']]['detected_language'].value_counts().head(3)
    top_non_eng = ', '.join([f"{lang}({cnt})" for lang, cnt in non_eng_langs.items()])

    domain_lang_analysis.append({
        'Domain': domain,
        'Total': total,
        'English': english,
        'Non-English': non_english,
        'English %': (english/total*100),
        'Non-English %': (non_english/total*100),
        'Top Non-English Languages': top_non_eng
    })

domain_lang_df = pd.DataFrame(domain_lang_analysis)
print(domain_lang_df.to_string(index=False))


# Detailed breakdown: Language distribution by Domain AND Class
print("="*60)
print("LANGUAGE DISTRIBUTION BY DOMAIN AND CLASS")
print("="*60)

detailed_analysis = []

for domain in DOMAINS:
    for label in [0, 1]:
        subset = df[(df['domain'] == domain) & (df['label'] == label)]

        if len(subset) == 0:
            continue

        total = len(subset)
        english = subset['is_english'].sum()

        # Get top 5 detected languages
        lang_dist = subset['detected_language'].value_counts().head(5).to_dict()

        detailed_analysis.append({
            'Domain': domain,
            'Class': 'Deceptive' if label == 1 else 'Non-Deceptive',
            'Total': total,
            'English': english,
            'English %': round(english/total*100, 2),
            'Non-English': total - english,
            'Non-English %': round((total-english)/total*100, 2),
            'Languages': lang_dist
        })

detailed_df = pd.DataFrame(detailed_analysis)
print(detailed_df[['Domain', 'Class', 'Total', 'English', 'English %', 'Non-English', 'Non-English %']].to_string(index=False))


# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Overall language distribution (top 10)
ax1 = axes[0, 0]
top_langs = df['detected_language'].value_counts().head(10)
colors = ['green' if lang == 'en' else 'coral' for lang in top_langs.index]
top_langs.plot(kind='bar', ax=ax1, color=colors)
ax1.set_title('Top 10 Detected Languages', fontsize=12)
ax1.set_xlabel('Language Code')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: English vs Non-English by domain
ax2 = axes[0, 1]
domain_lang_pivot = df.groupby('domain')['is_english'].agg(['sum', 'count'])
domain_lang_pivot['non_english'] = domain_lang_pivot['count'] - domain_lang_pivot['sum']
domain_lang_pivot[['sum', 'non_english']].plot(kind='bar', stacked=True, ax=ax2,
                                                color=['green', 'coral'])
ax2.set_title('English vs Non-English by Domain', fontsize=12)
ax2.set_xlabel('Domain')
ax2.set_ylabel('Count')
ax2.legend(['English', 'Non-English'])
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Non-English percentage by domain
ax3 = axes[1, 0]
non_eng_pct = domain_lang_df.set_index('Domain')['Non-English %']
non_eng_pct.plot(kind='bar', ax=ax3, color='coral')
ax3.set_title('Non-English Percentage by Domain', fontsize=12)
ax3.set_xlabel('Domain')
ax3.set_ylabel('Non-English %')
ax3.tick_params(axis='x', rotation=45)
ax3.axhline(y=non_eng_pct.mean(), color='red', linestyle='--', label=f'Mean: {non_eng_pct.mean():.1f}%')
ax3.legend()

# Plot 4: Language distribution by class
ax4 = axes[1, 1]
class_lang_pct.plot(kind='bar', ax=ax4, color=['green', 'coral'])
ax4.set_title('Language Distribution by Class', fontsize=12)
ax4.set_xlabel('Class')
ax4.set_ylabel('Percentage')
ax4.legend(['English', 'Non-English'])
ax4.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('language_distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'language_distribution_analysis.png'")


# =============================================================================
# 5. Create Dataset Splits (English-only vs Full)
# =============================================================================

# Create English-only and Full datasets
print("Creating dataset versions...\n")

# Full dataset (all languages)
df_full = df.copy()

# English-only dataset
df_english = df[df['is_english'] == True].copy()

print(f"Full dataset: {len(df_full):,} samples")
print(f"English-only dataset: {len(df_english):,} samples")
print(f"Samples removed: {len(df_full) - len(df_english):,} ({(1 - len(df_english)/len(df_full))*100:.2f}%)")

# Compare class distribution
print("\n--- Class Distribution Comparison ---")
print(f"Full - Deceptive: {df_full['label'].mean()*100:.2f}%")
print(f"English-only - Deceptive: {df_english['label'].mean()*100:.2f}%")


def prepare_train_test_data(df, test_size=0.2, random_state=42):
    """
    Prepare stratified train/test splits.
    Uses stratification to handle class imbalance.

    Source: scikit-learn train_test_split
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X = df['text'].values
    y = df['label'].values
    domains = df['domain'].values

    X_train, X_test, y_train, y_test, domains_train, domains_test = train_test_split(
        X, y, domains,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, domains_train, domains_test


# Prepare data for both versions
print("Preparing train/test splits...\n")

# Full dataset
X_train_full, X_test_full, y_train_full, y_test_full, domains_train_full, domains_test_full = \
    prepare_train_test_data(df_full)

# English-only dataset
X_train_eng, X_test_eng, y_train_eng, y_test_eng, domains_train_eng, domains_test_eng = \
    prepare_train_test_data(df_english)

print("Full Dataset:")
print(f"  Train: {len(X_train_full):,} | Test: {len(X_test_full):,}")
print(f"  Train class dist: {np.mean(y_train_full)*100:.2f}% deceptive")

print("\nEnglish-only Dataset:")
print(f"  Train: {len(X_train_eng):,} | Test: {len(X_test_eng):,}")
print(f"  Train class dist: {np.mean(y_train_eng)*100:.2f}% deceptive")


# =============================================================================
# 6. Traditional ML Classifiers (Random Forest & SVM)
# Source: scikit-learn - https://scikit-learn.org/
# =============================================================================

def create_tfidf_features(X_train, X_test, max_features=10000):
    """
    Create TF-IDF features from text data.

    Source: scikit-learn TfidfVectorizer
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,           # Minimum document frequency
        max_df=0.95,        # Maximum document frequency
        sublinear_tf=True   # Apply sublinear tf scaling
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer


print("Creating TF-IDF features...\n")

# Full dataset features
X_train_full_tfidf, X_test_full_tfidf, vectorizer_full = \
    create_tfidf_features(X_train_full, X_test_full)
print(f"Full dataset - TF-IDF shape: {X_train_full_tfidf.shape}")

# English-only features
X_train_eng_tfidf, X_test_eng_tfidf, vectorizer_eng = \
    create_tfidf_features(X_train_eng, X_test_eng)
print(f"English-only - TF-IDF shape: {X_train_eng_tfidf.shape}")


def train_and_evaluate_classifier(clf, X_train, X_test, y_train, y_test, clf_name, dataset_name):
    """
    Train classifier and return evaluation metrics.

    Uses metrics suitable for imbalanced datasets:
    - F1-Score (weighted and macro)
    - Balanced Accuracy
    - Precision and Recall

    Source: scikit-learn metrics
    """
    print(f"\nTraining {clf_name} on {dataset_name}...")
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Calculate metrics
    metrics = {
        'Classifier': clf_name,
        'Dataset': dataset_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred),
        'F1_Weighted': f1_score(y_test, y_pred, average='weighted'),
        'F1_Macro': f1_score(y_test, y_pred, average='macro'),
        'Precision_Weighted': precision_score(y_test, y_pred, average='weighted'),
        'Recall_Weighted': recall_score(y_test, y_pred, average='weighted')
    }

    print(f"  F1 (weighted): {metrics['F1_Weighted']:.4f}")
    print(f"  F1 (macro): {metrics['F1_Macro']:.4f}")
    print(f"  Balanced Accuracy: {metrics['Balanced_Accuracy']:.4f}")

    return metrics, y_pred, clf


# Train Random Forest
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

print("="*60)
print("RANDOM FOREST CLASSIFIER")
print("="*60)

rf_results = []

# Random Forest on Full Dataset
rf_full = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',  # Handle class imbalance
    random_state=SEED,
    n_jobs=-1
)
metrics_rf_full, pred_rf_full, _ = train_and_evaluate_classifier(
    rf_full, X_train_full_tfidf, X_test_full_tfidf,
    y_train_full, y_test_full,
    'Random Forest', 'Full (Multilingual)'
)
rf_results.append(metrics_rf_full)

# Random Forest on English-only Dataset
rf_eng = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1
)
metrics_rf_eng, pred_rf_eng, _ = train_and_evaluate_classifier(
    rf_eng, X_train_eng_tfidf, X_test_eng_tfidf,
    y_train_eng, y_test_eng,
    'Random Forest', 'English-only'
)
rf_results.append(metrics_rf_eng)


# Train SVM (Support Vector Machine)
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

print("="*60)
print("SVM CLASSIFIER")
print("="*60)

svm_results = []

# SVM on Full Dataset
svm_full = SVC(
    kernel='linear',
    C=1.0,
    class_weight='balanced',
    random_state=SEED
)
metrics_svm_full, pred_svm_full, _ = train_and_evaluate_classifier(
    svm_full, X_train_full_tfidf, X_test_full_tfidf,
    y_train_full, y_test_full,
    'SVM', 'Full (Multilingual)'
)
svm_results.append(metrics_svm_full)

# SVM on English-only Dataset
svm_eng = SVC(
    kernel='linear',
    C=1.0,
    class_weight='balanced',
    random_state=SEED
)
metrics_svm_eng, pred_svm_eng, _ = train_and_evaluate_classifier(
    svm_eng, X_train_eng_tfidf, X_test_eng_tfidf,
    y_train_eng, y_test_eng,
    'SVM', 'English-only'
)
svm_results.append(metrics_svm_eng)


# =============================================================================
# 7. Transformer-Based Classifier (DistilBERT)
# Source: HuggingFace Transformers - https://huggingface.co/docs/transformers/
# Model: distilbert-base-uncased - https://huggingface.co/distilbert-base-uncased
# =============================================================================

# DistilBERT Dataset Class
class FraudDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for fraud detection.
    Source: PyTorch Dataset API
    """
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """
    Compute metrics for HuggingFace Trainer.
    Uses metrics suitable for imbalanced data.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }


def train_distilbert(X_train, X_test, y_train, y_test, dataset_name, epochs=3, batch_size=16):
    """
    Train DistilBERT classifier.

    Source: HuggingFace Transformers
    Model: distilbert-base-uncased
    https://huggingface.co/distilbert-base-uncased
    """
    print(f"\n{'='*60}")
    print(f"Training DistilBERT on {dataset_name}")
    print(f"{'='*60}")

    # Load tokenizer and model
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # Create datasets
    train_dataset = FraudDataset(X_train, y_train, tokenizer)
    test_dataset = FraudDataset(X_test, y_test, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_{dataset_name.replace(" ", "_")}',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_weighted',
        seed=SEED
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()

    # Get predictions for detailed metrics
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)

    metrics = {
        'Classifier': 'DistilBERT',
        'Dataset': dataset_name,
        'Accuracy': eval_results['eval_accuracy'],
        'Balanced_Accuracy': eval_results['eval_balanced_accuracy'],
        'F1_Weighted': eval_results['eval_f1_weighted'],
        'F1_Macro': eval_results['eval_f1_macro'],
        'Precision_Weighted': eval_results['eval_precision'],
        'Recall_Weighted': eval_results['eval_recall']
    }

    print(f"\nResults for {dataset_name}:")
    print(f"  F1 (weighted): {metrics['F1_Weighted']:.4f}")
    print(f"  F1 (macro): {metrics['F1_Macro']:.4f}")
    print(f"  Balanced Accuracy: {metrics['Balanced_Accuracy']:.4f}")

    return metrics, y_pred, model


# Train DistilBERT on both datasets
# Note: This may take significant time depending on GPU availability

distilbert_results = []

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Sample size for faster training (optional - remove for full training)
# Comment out these lines for full dataset training
SAMPLE_SIZE = 5000  # Use smaller sample for demonstration
print(f"\nNote: Using sample of {SAMPLE_SIZE} for demonstration.")
print("Remove SAMPLE_SIZE limit for full training.\n")

# Sample data
np.random.seed(SEED)
sample_idx_full = np.random.choice(len(X_train_full), min(SAMPLE_SIZE, len(X_train_full)), replace=False)
sample_idx_eng = np.random.choice(len(X_train_eng), min(SAMPLE_SIZE, len(X_train_eng)), replace=False)

X_train_full_sample = X_train_full[sample_idx_full]
y_train_full_sample = y_train_full[sample_idx_full]

X_train_eng_sample = X_train_eng[sample_idx_eng]
y_train_eng_sample = y_train_eng[sample_idx_eng]


# Train on Full Dataset
metrics_bert_full, pred_bert_full, model_full = train_distilbert(
    X_train_full_sample, X_test_full[:1000],  # Smaller test set for speed
    y_train_full_sample, y_test_full[:1000],
    'Full (Multilingual)',
    epochs=2,
    batch_size=16
)
distilbert_results.append(metrics_bert_full)


# Train on English-only Dataset
metrics_bert_eng, pred_bert_eng, model_eng = train_distilbert(
    X_train_eng_sample, X_test_eng[:1000],
    y_train_eng_sample, y_test_eng[:1000],
    'English-only',
    epochs=2,
    batch_size=16
)
distilbert_results.append(metrics_bert_eng)


# =============================================================================
# 8. Results Comparison and Analysis
# =============================================================================

# Compile all results
all_results = rf_results + svm_results + distilbert_results
results_df = pd.DataFrame(all_results)

print("="*80)
print("OVERALL RESULTS COMPARISON")
print("="*80)
print(results_df.to_string(index=False))


# Calculate performance difference
print("\n" + "="*60)
print("PERFORMANCE DIFFERENCE (English-only vs Full)")
print("="*60)

for classifier in ['Random Forest', 'SVM', 'DistilBERT']:
    clf_results = results_df[results_df['Classifier'] == classifier]

    if len(clf_results) < 2:
        continue

    full_f1 = clf_results[clf_results['Dataset'].str.contains('Full')]['F1_Weighted'].values[0]
    eng_f1 = clf_results[clf_results['Dataset'].str.contains('English')]['F1_Weighted'].values[0]

    diff = eng_f1 - full_f1
    pct_change = (diff / full_f1) * 100

    print(f"\n{classifier}:")
    print(f"  Full dataset F1: {full_f1:.4f}")
    print(f"  English-only F1: {eng_f1:.4f}")
    print(f"  Difference: {diff:+.4f} ({pct_change:+.2f}%)")
    print(f"  Impact: {'Improved' if diff > 0 else 'Decreased'} with English-only data")


# Domain-wise performance analysis
# Note: This requires per-domain evaluation which we'll compute here

def evaluate_by_domain(y_true, y_pred, domains):
    """
    Calculate performance metrics for each domain.
    """
    domain_metrics = []

    for domain in DOMAINS:
        mask = domains == domain
        if mask.sum() == 0:
            continue

        y_true_domain = y_true[mask]
        y_pred_domain = y_pred[mask]

        domain_metrics.append({
            'Domain': domain,
            'Samples': mask.sum(),
            'Accuracy': accuracy_score(y_true_domain, y_pred_domain),
            'F1_Weighted': f1_score(y_true_domain, y_pred_domain, average='weighted', zero_division=0),
            'F1_Macro': f1_score(y_true_domain, y_pred_domain, average='macro', zero_division=0)
        })

    return pd.DataFrame(domain_metrics)


print("="*60)
print("DOMAIN-WISE PERFORMANCE (Random Forest - Full Dataset)")
print("="*60)
domain_perf_full = evaluate_by_domain(y_test_full, pred_rf_full, domains_test_full)
print(domain_perf_full.to_string(index=False))

print("\n" + "="*60)
print("DOMAIN-WISE PERFORMANCE (Random Forest - English-only)")
print("="*60)
domain_perf_eng = evaluate_by_domain(y_test_eng, pred_rf_eng, domains_test_eng)
print(domain_perf_eng.to_string(index=False))


# Aggregate metrics (Mean and Weighted)
print("="*60)
print("AGGREGATE PERFORMANCE METRICS")
print("="*60)

# Mean performance across domains
print("\n--- Mean Performance (unweighted average across domains) ---")
print(f"Full Dataset - Mean F1: {domain_perf_full['F1_Weighted'].mean():.4f}")
print(f"English-only - Mean F1: {domain_perf_eng['F1_Weighted'].mean():.4f}")

# Weighted performance (weighted by number of samples)
print("\n--- Weighted Performance (weighted by domain size) ---")
weighted_f1_full = np.average(
    domain_perf_full['F1_Weighted'],
    weights=domain_perf_full['Samples']
)
weighted_f1_eng = np.average(
    domain_perf_eng['F1_Weighted'],
    weights=domain_perf_eng['Samples']
)
print(f"Full Dataset - Weighted F1: {weighted_f1_full:.4f}")
print(f"English-only - Weighted F1: {weighted_f1_eng:.4f}")


# Visualization of results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: F1 Score comparison by classifier
ax1 = axes[0, 0]
classifiers = results_df['Classifier'].unique()
x = np.arange(len(classifiers))
width = 0.35

full_f1 = [results_df[(results_df['Classifier']==c) & (results_df['Dataset'].str.contains('Full'))]['F1_Weighted'].values[0]
           if len(results_df[(results_df['Classifier']==c) & (results_df['Dataset'].str.contains('Full'))]) > 0 else 0
           for c in classifiers]
eng_f1 = [results_df[(results_df['Classifier']==c) & (results_df['Dataset'].str.contains('English'))]['F1_Weighted'].values[0]
          if len(results_df[(results_df['Classifier']==c) & (results_df['Dataset'].str.contains('English'))]) > 0 else 0
          for c in classifiers]

bars1 = ax1.bar(x - width/2, full_f1, width, label='Full (Multilingual)', color='coral')
bars2 = ax1.bar(x + width/2, eng_f1, width, label='English-only', color='green')
ax1.set_ylabel('F1 Score (Weighted)')
ax1.set_title('F1 Score by Classifier and Dataset')
ax1.set_xticks(x)
ax1.set_xticklabels(classifiers)
ax1.legend()
ax1.set_ylim(0, 1)

# Plot 2: Domain-wise F1 comparison
ax2 = axes[0, 1]
x = np.arange(len(DOMAINS))
ax2.bar(x - width/2, domain_perf_full['F1_Weighted'], width, label='Full', color='coral')
ax2.bar(x + width/2, domain_perf_eng['F1_Weighted'], width, label='English-only', color='green')
ax2.set_ylabel('F1 Score (Weighted)')
ax2.set_title('Domain-wise F1 Score (Random Forest)')
ax2.set_xticks(x)
ax2.set_xticklabels([d.replace('_', '\n') for d in DOMAINS], fontsize=8)
ax2.legend()

# Plot 3: Confusion Matrix (Full Dataset)
ax3 = axes[1, 0]
cm_full = confusion_matrix(y_test_full, pred_rf_full)
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_title('Confusion Matrix - Full Dataset (RF)')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# Plot 4: Confusion Matrix (English-only)
ax4 = axes[1, 1]
cm_eng = confusion_matrix(y_test_eng, pred_rf_eng)
sns.heatmap(cm_eng, annot=True, fmt='d', cmap='Greens', ax=ax4)
ax4.set_title('Confusion Matrix - English-only (RF)')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'classification_results.png'")


# =============================================================================
# 9. Summary and Conclusions
# =============================================================================

# Generate summary report
print("="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

print("\n### Dataset Analysis ###")
print(f"Total samples analyzed: {len(df):,}")
print(f"English samples: {df['is_english'].sum():,} ({df['is_english'].mean()*100:.2f}%)")
print(f"Non-English samples: {(~df['is_english']).sum():,} ({(~df['is_english']).mean()*100:.2f}%)")
print(f"Unique languages detected: {df['detected_language'].nunique()}")

print("\n### Language Distribution by Domain ###")
print(domain_lang_df[['Domain', 'Total', 'Non-English', 'Non-English %']].to_string(index=False))

print("\n### Classification Performance Summary ###")
print(results_df[['Classifier', 'Dataset', 'F1_Weighted', 'Balanced_Accuracy']].to_string(index=False))

print("\n### Hypothesis Testing Results ###")
print("H1 (Data Composition): ", end="")
non_eng_pct = (~df['is_english']).mean() * 100
if non_eng_pct > 1:
    print(f"SUPPORTED - {non_eng_pct:.2f}% non-English content found")
else:
    print(f"NOT SUPPORTED - Only {non_eng_pct:.2f}% non-English content")

print("H2 (Performance Impact): ", end="")
# Compare best F1 scores
if len(results_df) > 0:
    full_best = results_df[results_df['Dataset'].str.contains('Full')]['F1_Weighted'].max()
    eng_best = results_df[results_df['Dataset'].str.contains('English')]['F1_Weighted'].max()
    if eng_best > full_best:
        print(f"SUPPORTED - English-only shows higher F1 ({eng_best:.4f} vs {full_best:.4f})")
    else:
        print(f"NOT SUPPORTED - Full dataset shows comparable/better F1 ({full_best:.4f} vs {eng_best:.4f})")


# Save results to CSV
results_df.to_csv('classification_results.csv', index=False)
domain_lang_df.to_csv('language_distribution_by_domain.csv', index=False)

# Save detailed language analysis
df[['text', 'label', 'domain', 'detected_language', 'language_confidence', 'is_english']].to_csv(
    'difraud_language_analysis.csv', index=False
)

print("\nResults saved to:")
print("  - classification_results.csv")
print("  - language_distribution_by_domain.csv")
print("  - difraud_language_analysis.csv")


# =============================================================================
# References and Sources
# =============================================================================
"""
References and Sources:

Dataset:
- DIFrauD Dataset: Boumber, D., et al. (2024). "Domain-Agnostic Adapter Architecture
  for Deception Detection." LREC-COLING 2024.
  Available at: https://huggingface.co/datasets/difraud/difraud

Libraries and Code Sources:
- langdetect: Language detection library (port of Google's language-detection).
  https://pypi.org/project/langdetect/
- HuggingFace datasets: Dataset loading library.
  https://huggingface.co/docs/datasets/
- HuggingFace transformers: Transformer models (DistilBERT).
  https://huggingface.co/docs/transformers/
- scikit-learn: ML classifiers (Random Forest, SVM) and metrics.
  https://scikit-learn.org/
- DistilBERT model: distilbert-base-uncased.
  https://huggingface.co/distilbert-base-uncased

Academic References:
- Conneau, A., et al. (2020). "Unsupervised cross-lingual representation learning
  at scale." ACL 2020.
- Devlin, J., et al. (2019). "BERT: Pre-training of deep bidirectional transformers."
  NAACL 2019.
- Verma, R. M., et al. (2019). "Data quality for security challenges." ACM CCS 2019.

Metrics Choice Justification:
- F1-Score (Weighted): Used as primary metric due to class imbalance in DIFrauD dataset.
  Weighted F1 accounts for class distribution.
- F1-Score (Macro): Unweighted average across classes, useful for evaluating performance
  on minority class.
- Balanced Accuracy: Accounts for class imbalance by averaging recall across classes.

Code Notes:
- All code in this notebook is original unless otherwise noted
- API usage follows official documentation from respective libraries
- Random seed (42) used for reproducibility
"""
