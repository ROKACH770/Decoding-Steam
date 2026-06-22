import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm.auto import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ==========================================
# 1. Data Preprocessing & Stratified Sampling
# ==========================================

def clean_text(text: str) -> str:
    """
    Cleans raw review text by removing special characters, URLs, and extra spaces.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces
    return text.lower()


def prepare_data(file_path: str, samples_per_class: int = 2500) -> pd.DataFrame:
    """
    Loads, preprocesses, and performs STRATIFIED undersampling.
    Balances the classes (positive/negative) while maintaining the true probabilistic
    distribution of player engagement (playtime).
    """
    cols_to_use = [
        'review', 'recommended', 'votes_helpful',
        'author.playtime_at_review', 'steam_purchase', 'author.num_games_owned'
    ]

    print("Reading data chunk for stratified sampling...")
    df_chunk = pd.read_csv(file_path, usecols=cols_to_use, nrows=500000)
    df_chunk = df_chunk.dropna(subset=['review', 'recommended', 'author.playtime_at_review'])

    # Create strata (layers) based on playtime distribution (Quartiles)
    df_chunk['playtime_stratum'] = pd.qcut(df_chunk['author.playtime_at_review'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Calculate the probabilistic weights of each stratum in the overall dataset
    stratum_proportions = df_chunk['playtime_stratum'].value_counts(normalize=True)

    sampled_dfs = []

    # Sample proportionally from each stratum to maintain real-world distribution
    for stratum, proportion in stratum_proportions.items():
        n_samples_stratum = int(samples_per_class * proportion)

        df_stratum_pos = df_chunk[(df_chunk['playtime_stratum'] == stratum) & (df_chunk['recommended'] == True)]
        df_stratum_neg = df_chunk[(df_chunk['playtime_stratum'] == stratum) & (df_chunk['recommended'] == False)]

        actual_samples = min(n_samples_stratum, len(df_stratum_pos), len(df_stratum_neg))

        if actual_samples > 0:
            sampled_dfs.append(df_stratum_pos.sample(n=actual_samples, random_state=42))
            sampled_dfs.append(df_stratum_neg.sample(n=actual_samples, random_state=42))

    # Combine all stratified samples into one balanced DataFrame
    df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Stratified sample ready. Total size: {len(df)} rows.")

    # Apply text cleaning and fill missing metadata
    df['clean_review'] = df['review'].apply(clean_text)
    df['author.num_games_owned'] = df['author.num_games_owned'].fillna(df['author.num_games_owned'].median())
    df['steam_purchase'] = df['steam_purchase'].fillna(True).astype(int)
    df['recommended'] = df['recommended'].astype(int)

    return df


# ==========================================
# 2. Baseline Model (TF-IDF + Logistic Regression)
# ==========================================

def run_baseline_model(df: pd.DataFrame) -> None:
    """
    Trains and evaluates a baseline Logistic Regression model using TF-IDF features.
    Predicts the 'recommended' status.
    """
    print("\n--- Running Baseline Model (TF-IDF + Logistic Regression) ---")

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'], df['recommended'], test_size=0.2, random_state=42
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Model
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_tfidf, y_train)

    # Predictions & Evaluation
    y_pred = clf.predict(X_test_tfidf)
    print("\nBaseline Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Rec', 'Rec'], yticklabels=['Not Rec', 'Rec'])
    plt.title('Baseline Model: Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. Transformer + Metadata Fusion Architecture
# ==========================================

class SteamReviewDataset(Dataset):
    """
    Custom PyTorch Dataset for loading text and metadata features.
    """

    def __init__(self, texts, metadata, targets, tokenizer, max_len=128):
        self.texts = texts
        self.metadata = metadata
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        meta = self.metadata[item]
        target = self.targets[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': torch.tensor(meta, dtype=torch.float),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class HybridTransformerModel(nn.Module):
    """
    A unified architecture that leverages a pre-trained Transformer for self-attention
    text representations, concatenated with reviewer metadata to predict recommendations.
    """

    def __init__(self, transformer_name: str, n_meta_features: int, n_classes: int):
        super(HybridTransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)

        # Freeze transformer parameters to speed up training on standard hardware
        for param in self.transformer.parameters():
            param.requires_grad = False

        transformer_out_dim = self.transformer.config.hidden_size

        # Combined dense layers
        self.fc1 = nn.Linear(transformer_out_dim + n_meta_features, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, n_classes)

    def forward(self, input_ids, attention_mask, metadata):
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the [CLS] token representation
        text_features = transformer_output.last_hidden_state[:, 0, :]

        # Concatenate textual features with reviewer metadata
        combined_features = torch.cat((text_features, metadata), dim=1)

        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.out(x)
        return logits


def train_hybrid_model(df: pd.DataFrame) -> None:
    """
    Initializes and trains the Transformer-Metadata hybrid model.
    Includes data loaders, loss function, and basic evaluation loop.
    """
    print("\n--- Setting up Hybrid Transformer Model ---")

    # Prepare metadata features (Normalization)
    meta_cols = ['author.playtime_at_review', 'steam_purchase', 'author.num_games_owned']
    for col in meta_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    X_text = df['clean_review'].values
    X_meta = df[meta_cols].values
    y = df['recommended'].values

    xt_train, xt_test, xm_train, xm_test, y_train, y_test = train_test_split(
        X_text, X_meta, y, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = SteamReviewDataset(xt_train, xm_train, y_train, tokenizer)
    test_dataset = SteamReviewDataset(xt_test, xm_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HybridTransformerModel('distilbert-base-uncased', n_meta_features=3, n_classes=2)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop (1 epoch for demonstration)
    print("Training model (1 Epoch)...")
    model.train()
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, metadata)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}\n")

    # Evaluation
    print("\nEvaluating Hybrid Model...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids, attention_mask, metadata)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print("\nHybrid Model Classification Report:")
    print(classification_report(all_targets, all_preds))

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Not Rec', 'Rec'], yticklabels=['Not Rec', 'Rec'])
    plt.title('Hybrid Transformer: Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ensure this points to your CSV file
    dataset_path = "steam_reviews_english.csv"

    print("Loading and preprocessing data...")
    df = prepare_data(dataset_path, samples_per_class=2500)

    # Run Baseline Evaluation
    run_baseline_model(df)

    # Run Transformer + Metadata Evaluation
    train_hybrid_model(df)

    print("Execution finished.")