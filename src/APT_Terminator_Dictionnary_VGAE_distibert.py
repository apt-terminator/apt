import os
import argparse
import matplotlib.pyplot as plt
import openai
import time
import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
import numpy as np
from openai import OpenAI
import tiktoken
import pandas as pd
import numpy as np
import ot
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
# Example embeddings generation
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import torch
from sentence_transformers import SentenceTransformer
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import Normalize
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AlbertTokenizer, AlbertModel
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
import numpy as np
import random
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import sys
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics import classification_report
# 1. One-Class SVM
import numpy as np
from sklearn.svm import OneClassSVM
from transformers import BartTokenizer, BartModel
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.layers import Reshape
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, ndcg_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_roc_curves(y_true, recon_errors, model_names):
    """
    Plot ROC curves for multiple models (AE, VAE, DAE).
    
    Parameters:
    -----------
    y_true : numpy array
        Ground truth binary labels (0 = normal, 1 = anomaly).
    recon_errors : dict
        A dictionary where keys are model names and values are reconstruction errors.
    model_names : list
        A list of model names for labeling the ROC curves.
    """
    plt.figure(figsize=(10, 7))
    
    for model_name in model_names:
        # Compute the ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, recon_errors[model_name])
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    # Plot settings
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    plt.title('ROC Curves for AE, VAE, and DAE')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.4)
    plt.show()

def sampling(args):
    """Reparameterization trick: sample z ~ N(mu, sigma^2)."""
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(0.5 * log_var) * epsilon



def build_autoencoder_with_attention_fixed(input_dim, num_heads=2, attention_dim=64):
    input_layer = Input(shape=(input_dim,))
    
    # ENCODER
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)

    # Reshape input for MultiHeadAttention
    attention_input = Dense(attention_dim, activation='relu')(encoded)  # Project input for attention
    attention_input = Reshape((1, attention_dim))(attention_input)  # Reshape to [batch_size, seq_length=1, attention_dim]

    # Attention Layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)(attention_input, attention_input)
    attention_output = Reshape((attention_dim,))(attention_output)  # Flatten back to 2D
    attention_output = LayerNormalization()(attention_output + attention_input[:, 0, :])  # Residual connection
    attention_output = Dropout(0.1)(attention_output)

    # Further compress
    encoded = Dense(32, activation='relu')(attention_output)
    encoded = Dense(16, activation='relu')(encoded)

    # DECODER
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Models
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


from tensorflow.keras.layers import GaussianNoise, Reshape

def build_denoising_autoencoder_with_attention_fixed(input_dim, noise_factor=0.1, num_heads=2, attention_dim=64):
    input_layer = Input(shape=(input_dim,))
    
    # Add Noise
    noisy_input = GaussianNoise(noise_factor)(input_layer)
    
    # ENCODER
    encoded = Dense(128, activation='relu')(noisy_input)
    encoded = Dense(64, activation='relu')(encoded)

    # Reshape input for MultiHeadAttention
    attention_input = Dense(attention_dim, activation='relu')(encoded)
    attention_input = Reshape((1, attention_dim))(attention_input)

    # Attention Layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)(attention_input, attention_input)
    attention_output = Reshape((attention_dim,))(attention_output)
    attention_output = LayerNormalization()(attention_output + attention_input[:, 0, :])  # Residual connection
    attention_output = Dropout(0.1)(attention_output)

    # Further compress
    encoded = Dense(32, activation='relu')(attention_output)
    encoded = Dense(16, activation='relu')(encoded)

    # DECODER
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Models
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

def sampling(args):
    """Reparameterization trick: sample z ~ N(mu, sigma^2)."""
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(0.5 * log_var) * epsilon

def build_variational_autoencoder_with_attention_fixed(input_dim, latent_dim=16, num_heads=2, attention_dim=64):
    input_layer = Input(shape=(input_dim,))
    
    # ENCODER
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)

    # Reshape input for MultiHeadAttention
    attention_input = Dense(attention_dim, activation='relu')(encoded)
    attention_input = Reshape((1, attention_dim))(attention_input)

    # Attention Layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)(attention_input, attention_input)
    attention_output = Reshape((attention_dim,))(attention_output)
    attention_output = LayerNormalization()(attention_output + attention_input[:, 0, :])  # Residual connection
    attention_output = Dropout(0.1)(attention_output)

    # Latent Variables
    mu = Dense(latent_dim, activation='linear')(attention_output)
    log_var = Dense(latent_dim, activation='linear')(attention_output)
    z = Lambda(sampling)([mu, log_var])

    # DECODER
    decoded = Dense(64, activation='relu')(z)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Models
    vae = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=mu)

    # Loss Function
    reconstruction_loss = K.mean(K.square(input_layer - decoded))
    kl_loss = -0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var))
    vae_loss = reconstruction_loss + kl_loss

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder

def build_baseline_autoencoder_with_attention(input_dim, num_heads=2, attention_dim=64):
    input_layer = Input(shape=(input_dim,))
    
    # ENCODER
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)

    # Attention Layer
    attention_input = Dense(attention_dim, activation='relu')(encoded)  # Project input for attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)(attention_input, attention_input)
    attention_output = LayerNormalization()(attention_output + attention_input)  # Residual connection
    attention_output = Dropout(0.1)(attention_output)

    # Further compress
    encoded = Dense(32, activation='relu')(attention_output)
    encoded = Dense(16, activation='relu')(encoded)

    # DECODER
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Models
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def build_variational_autoencoder_with_attention(input_dim, latent_dim=16, num_heads=2, attention_dim=64):
    input_layer = Input(shape=(input_dim,))
    
    # ENCODER
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)

    # Attention Layer
    attention_input = Dense(attention_dim, activation='relu')(encoded)  # Project input for attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)(attention_input, attention_input)
    attention_output = LayerNormalization()(attention_output + attention_input)  # Residual connection
    attention_output = Dropout(0.1)(attention_output)

    # Latent Variables
    mu = Dense(latent_dim, activation='linear')(attention_output)
    log_var = Dense(latent_dim, activation='linear')(attention_output)
    z = Lambda(sampling)([mu, log_var])

    # DECODER
    decoded = Dense(64, activation='relu')(z)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Models
    vae = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=mu)

    # Loss Function
    reconstruction_loss = K.mean(K.square(input_layer - decoded))
    kl_loss = -0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var))
    vae_loss = reconstruction_loss + kl_loss

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder


def build_denoising_autoencoder_with_attention(input_dim, noise_factor=0.1, num_heads=2, attention_dim=64):
    input_layer = Input(shape=(input_dim,))
    
    # Add Noise
    noisy_input = GaussianNoise(noise_factor)(input_layer)
    
    # ENCODER
    encoded = Dense(128, activation='relu')(noisy_input)
    encoded = Dense(64, activation='relu')(encoded)

    # Attention Layer
    attention_input = Dense(attention_dim, activation='relu')(encoded)  # Project input for attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)(attention_input, attention_input)
    attention_output = LayerNormalization()(attention_output + attention_input)  # Residual connection
    attention_output = Dropout(0.1)(attention_output)

    # Further compress
    encoded = Dense(32, activation='relu')(attention_output)
    encoded = Dense(16, activation='relu')(encoded)

    # DECODER
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Models
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def build_variational_autoencoder(input_dim, latent_dim=16):
    input_layer = Input(shape=(input_dim,))
    
    # ENCODER
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    mu = Dense(latent_dim, activation='linear')(encoded)
    log_var = Dense(latent_dim, activation='linear')(encoded)
    z = Lambda(sampling)([mu, log_var])  # Latent vector z

    # DECODER
    decoded = Dense(64, activation='relu')(z)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Models
    vae = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=mu)

    # Loss function
    reconstruction_loss = K.mean(K.square(input_layer - decoded))
    kl_loss = -0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var))
    vae_loss = reconstruction_loss + kl_loss

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder


def plot_multiple_roc_curves(df, label_column, score_columns_dict):
    """
    Plot multiple ROC curves in the same figure, comparing different embeddings or models.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'label_column' and the columns for various scores.
    label_column : str
        The column name in df that contains the ground-truth labels (0 = normal, 1 = anomaly).
    score_columns_dict : dict
        A dictionary mapping a descriptive name (e.g., 'BERT', 'ALBERT') to the column name
        in df that holds the anomaly score for that model.
        
        Example: {
            'BERT' : 'recon_error_embedding_bert',
            'ALBERT': 'recon_error_embedding_albert',
            ...
        }
    """
    y_true = df[label_column].values

    plt.figure(figsize=(8, 8))

    # Plot one ROC curve per score column
    for model_name, score_column in score_columns_dict.items():
        y_scores = df[score_column].values
        
        # Compute the ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    # Plot the random guess line
 

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison for Multiple LLM Embeddings')
    plt.legend(loc='lower right')
    plt.show()




def plot_multiple_roc_curves2(df, label_column, score_columns_dict):
    """
    Plot multiple ROC curves in the same figure, comparing different embeddings or models.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'label_column' and the columns for various scores.
    label_column : str
        The column name in df that contains the ground-truth labels (0 = normal, 1 = anomaly).
    score_columns_dict : dict
        A dictionary mapping a descriptive name (e.g., 'BERT', 'ALBERT') to the column name
        in df that holds the anomaly score for that model.
        
        Example: {
            'BERT' : 'recon_error_embedding_bert',
            'ALBERT': 'recon_error_embedding_albert',
            ...
        }
    """
    y_true = df[label_column].values

    plt.figure(figsize=(8, 6))

    # Plot one ROC curve per score column
    for model_name, score_column in score_columns_dict.items():
        y_scores = df[score_column].values
        
        # Compute the ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    # Plot the random guess line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison for Multiple LLM Embeddings')

    # Place the legend outside the plot
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.05, 1),  # Adjust to place the legend outside
        fontsize='small',          # Adjust font size
        frameon=False              # Remove box around legend
    )

    #plt.tight_layout()  # Adjust the layout to avoid overlap
    plt.show()

def detect_with_ocsvm_score(X_train_normal, X_test, y_test):
    """
    Perform anomaly detection using OC-SVM and calculate AUC.

    Parameters:
    -----------
    X_train_normal : numpy array or pandas DataFrame
        Training data consisting only of normal samples.
    X_test : numpy array or pandas DataFrame
        Test data containing both normal and anomalous samples.
    y_test : numpy array or pandas Series
        Ground truth labels for the test data (0 = normal, 1 = anomaly).

    Returns:
    --------
    ocsvm_scores : numpy array
        Anomaly scores for each sample in X_test.
    auc : float
        AUC score for the test set.
    """
    # Initialize One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    
    # Fit the model on normal training data
    ocsvm.fit(X_train_normal)
    
    # Compute anomaly scores for test data
    ocsvm_scores = -ocsvm.decision_function(X_test)
    
    # Calculate AUC
    auc = roc_auc_score(y_test, ocsvm_scores)
    
    return ocsvm_scores, auc


def compute_ndcg(y_true, anomaly_scores, k=None):
    """
    Compute NDCG for the anomaly detection ranking.
    - y_true: array of ground-truth labels (0 = normal, 1 = anomaly)
    - anomaly_scores: continuous scores (higher => more anomalous)
    - k: optional cutoff; if None, uses the entire list
    """
    # sklearn's ndcg_score expects shape (1, n_samples) for each argument
    return ndcg_score([y_true], [anomaly_scores], k=k)



# Function to calculate NDCG
def calculate_ndcg(df, label_column, score_column, k):
    true_relevance = df[label_column].values
    scores = df[score_column].values
    return ndcg_score([true_relevance], [scores], k=k)

def calculate_ndcg_full(df, label_column, score_column):

    true_relevance = df[label_column].values
    predicted_scores = df[score_column].values
    # ndcg_score expects arrays of shape (1, n_samples)
    # Passing k=None (or omitting it) uses the full ranking
    ndcg_value = ndcg_score(
        [true_relevance],
        [predicted_scores]
    )
    return ndcg_value



def plot_roc_curve(df, label_column, score_column):
    """
    Plots an ROC curve given a DataFrame with true labels and anomaly scores.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing at least 'label_column' and 'score_column'.
    label_column : str
        Name of the column containing binary labels (0 for normal, 1 for anomaly).
    score_column : str
        Name of the column containing the anomaly score (e.g., reconstruction error).
    """
    y_true = df[label_column].values
    y_scores = df[score_column].values

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange',
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


def build_autoencoder_with_attention(input_dim, num_heads=2, attention_dim=64):
    input_layer = Input(shape=(input_dim,))
    
    # ENCODER
    # Dense layers to reduce dimensions
    encoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0015))(input_layer)
    encoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0015))(encoded)
    
    # Self-Attention Layer
    attention_input = Dense(attention_dim, activation="relu")(encoded)  # Project input for attention
    attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)(attention_input, attention_input)
    attention_output = LayerNormalization()(attention_layer + attention_input)  # Residual connection + normalization
    attention_output = Dropout(0.1)(attention_output)
    
    # Further compress
    encoded = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0015))(attention_output)
    encoded = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0015))(encoded)
    
    # DECODER
    decoded = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0015))(encoded)
    decoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0015))(decoded)
    decoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0015))(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Models
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def build_denoising_autoencoder(input_dim, noise_factor=0.1):
    input_layer = Input(shape=(input_dim,))
    noisy_input = GaussianNoise(noise_factor)(input_layer)
    
    encoded = Dense(128, activation='relu')(noisy_input)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def build_autoencoder2(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # ENCODER
    encoded = Dense(  128,     activation='relu',        kernel_regularizer=regularizers.l2(0.0015) )(input_layer)
    encoded = Dense(    64,   activation='relu',    kernel_regularizer=regularizers.l2(0.0015) )(encoded)
    
    encoded = Dense(    32,        activation='relu',        kernel_regularizer=regularizers.l2(0.0015)    )(encoded)
    
    encoded = Dense(   16,   activation='relu',  kernel_regularizer=regularizers.l2(0.0015))(encoded)

    # DECODER
    decoded = Dense( 32,  activation='relu', kernel_regularizer=regularizers.l2(0.0015) )(encoded)
    decoded = Dense( 64, activation='relu',  kernel_regularizer=regularizers.l2(0.0015))(decoded)
    decoded = Dense( 128,  activation='relu', kernel_regularizer=regularizers.l2(0.0015) )(decoded)
    decoded = Dense( input_dim,  activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    # Note: 'encoded' here is the final encoder layer
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Autoencoder for anomaly detection
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)


    decoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def detect_with_ocsvm(X_clean: np.ndarray, X_noisy: np.ndarray):
    """
    Train a One-Class SVM on normal data (X_clean) and
    predict anomalies in X_noisy.

    Returns a list of 0/1 predictions, where 1 indicates anomaly.
    """
    from sklearn.svm import OneClassSVM

    # Create and train the model on normal data only
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)  # tweak 'nu' as needed
    ocsvm.fit(X_clean)

    # Predict on X_noisy; scikit-learn returns +1 for inliers and -1 for outliers
    predictions = ocsvm.predict(X_noisy)

    # Convert [-1, +1] to [1, 0] for anomaly vs. normal
    anomalies = [1 if p == -1 else 0 for p in predictions]
    return anomalies



def detect_with_dbscan_score(X_train_normal, X_test, y_test, eps=0.5, min_samples=5):
    """
    Perform anomaly detection using DBSCAN and calculate AUC.

    Parameters:
    -----------
    X_train_normal : numpy array or pandas DataFrame
        Training data consisting only of normal samples (used for fitting DBSCAN).
    X_test : numpy array or pandas DataFrame
        Test data containing both normal and anomalous samples.
    y_test : numpy array or pandas Series
        Ground truth labels for the test data (0 = normal, 1 = anomaly).
    eps : float, optional (default=0.5)
        Maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, optional (default=5)
        Minimum number of samples in a neighborhood to form a dense region.

    Returns:
    --------
    dbscan_scores : numpy array
        Anomaly scores for each sample in X_test (higher score = more anomalous).
    auc : float
        AUC score for the test set.
    """
    # Fit DBSCAN on training data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_train_normal)
    
    # Assign cluster labels to the test data
    test_labels = dbscan.fit_predict(X_test)
    
    # Compute anomaly scores
    dbscan_scores = np.where(test_labels == -1, 1.0, 0.0)  # 1 for anomaly, 0 for normal

    # Calculate AUC
    auc = roc_auc_score(y_test, dbscan_scores)
    
    return dbscan_scores, auc


def detect_with_isolation_forest_score(X_train_normal, X_test, y_test):
    """
    Perform anomaly detection using Isolation Forest and calculate AUC.

    Parameters:
    -----------
    X_train_normal : numpy array or pandas DataFrame
        Training data consisting only of normal samples.
    X_test : numpy array or pandas DataFrame
        Test data containing both normal and anomalous samples.
    y_test : numpy array or pandas Series
        Ground truth labels for the test data (0 = normal, 1 = anomaly).

    Returns:
    --------
    isolation_scores : numpy array
        Anomaly scores for each sample in X_test.
    auc : float
        AUC score for the test set.
    """
    # Initialize Isolation Forest
    isolation_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.1,
        random_state=42
    )
    
    # Fit the model on normal training data
    isolation_forest.fit(X_train_normal)
    
    # Compute anomaly scores for test data
    isolation_scores = -isolation_forest.decision_function(X_test)
    
    # Calculate AUC
    auc = roc_auc_score(y_test, isolation_scores)
    
    return isolation_scores, auc


def detect_with_isolation_forest(X_clean: np.ndarray, X_noisy: np.ndarray):
    """
    Train an Isolation Forest on normal data (X_clean) and
    predict anomalies in X_noisy.

    Returns a list of 0/1 predictions, where 1 indicates anomaly.
    """
    from sklearn.ensemble import IsolationForest

    # Initialize Isolation Forest with a small contamination rate
    # if you have an estimate of anomaly prevalence, adjust 'contamination'
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_clean)

    # Predict on X_noisy; returns +1 for inliers and -1 for outliers
    predictions = iso_forest.predict(X_noisy)

    # Convert [-1, +1] to [1, 0] for anomaly vs. normal
    anomalies = [1 if p == -1 else 0 for p in predictions]
    return anomalies


def detect_with_dbscan(X_noisy: np.ndarray):
    """
    Use DBSCAN clustering on X_noisy directly and treat
    all points in the '-1' cluster as anomalies.

    Since DBSCAN doesn't require training data, we assume X_noisy 
    has both normal and potential anomalies. This function treats 
    any point DBSCAN deems as noise (label=-1) as anomaly.

    Returns a list of 0/1 predictions, where 1 indicates anomaly.
    """
    from sklearn.cluster import DBSCAN

    # Run DBSCAN; adjust eps, min_samples to match your data density
    db = DBSCAN(eps=0.5, min_samples=5)
    labels = db.fit_predict(X_noisy)

    # DBSCAN assigns -1 to outliers (noise)
    anomalies = [1 if label == -1 else 0 for label in labels]
    return anomalies


def add_gaussian_noise(X, noise_factor=0.1):
    """
    Adds Gaussian noise to each embedding vector.
    
    Parameters
    ----------
    X : numpy.ndarray
        Original embeddings of shape (n_samples, embedding_dim).
    noise_factor : float
        Magnitude of noise to add.

    Returns
    -------
    X_noisy : numpy.ndarray
        Noisy embeddings of the same shape.
    """
    noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    return np.clip(noisy, 0., 1.)  # If your data are scaled between 0 and 1


# Define a function to create sentences for processes
def process_to_sentence(row):
    active_events = [event for event, value in row[event_labels].items() if value == 1]
    return f"Process {row.name} has {', '.join(active_events)}."

 # Generate embeddings for each LLM
def generate_embeddings(sentences, model):
    return [model.encode(sentence).tolist() for sentence in sentences]

# Function to visualize embeddings using t-SNE
def visualize_embeddings(df, embedding_column, labels_column):
    embeddings = np.array(df[embedding_column].tolist())
    labels = df[labels_column]

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    for label in labels.unique():
        idx = labels == label
        if idx ==0:
            _lbl="Normal"
        else:
             _lbl="Anomaly"
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=f"Label {label}", alpha=0.7)

    plt.title(f"t-SNE Visualization of {embedding_column}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()


# Function to visualize embeddings using t-SNE
def visualize_embeddings2(df, embedding_column, labels_column):
    embeddings = np.array(df[embedding_column].tolist())
    labels = df[labels_column]

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    for label in labels.unique():
        idx = labels == label
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=f"Label {label}", alpha=0.7)

    plt.title(f"t-SNE Visualization of {embedding_column}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()

def visualize_embeddings_3d_v1(df, embedding_column, labels_column):
    """
    Perform a 3D t-SNE on the embeddings and visualize them using matplotlib.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the embeddings and labels.
    embedding_column : str
        The column name in df that contains the embeddings (list or numpy array).
    labels_column : str
        The column name in df that contains labels (e.g., 0 = normal, 1 = anomaly).
    """
    # Convert the list of embeddings to a numpy array
    embeddings = np.array(df[embedding_column].tolist())
    labels = df[labels_column]

    # Create a t-SNE object with 3 components
    tsne = TSNE(n_components=3, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Set up a 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each label group in a different color
    for label_val in labels.unique():
        idx = labels == label_val
        if label_val == 0:
            lbl = "Normal"
        else:
            lbl = "Anomaly"
        ax.scatter(
            reduced_embeddings[idx, 0],
            reduced_embeddings[idx, 1],
            reduced_embeddings[idx, 2],
            label=lbl,
            alpha=0.7
        )

    ax.set_title(f"3D t-SNE Visualization of Linux Bovia PE using {embedding_column}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend()
    plt.show()

def visualize_embeddings_3d(df, embedding_column, labels_column):
    """
    Perform a 3D t-SNE on the embeddings and visualize them using matplotlib.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the embeddings and labels.
    embedding_column : str
        The column name in df that contains the embeddings (list or numpy array).
    labels_column : str
        The column name in df that contains labels (e.g., 0 = normal, 1 = anomaly).
    """
    # Convert the list of embeddings to a numpy array
    embeddings = np.array(df[embedding_column].tolist())
    labels = df[labels_column]

    # Create a t-SNE object with 3 components
    tsne = TSNE(n_components=3, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Separate the normal vs. anomaly data
    normal_idx = labels == 0
    anomaly_idx = labels == 1

    normal_emb = reduced_embeddings[normal_idx]
    anomaly_emb = reduced_embeddings[anomaly_idx]

    # Set up a 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot normal points first
    ax.scatter(
        normal_emb[:, 0],
        normal_emb[:, 1],
        normal_emb[:, 2],
        c='blue',
        alpha=0.2,
        label='Normal',
        zorder=2
    )

    # Plot anomaly points on top
    ax.scatter(
        anomaly_emb[:, 0],
        anomaly_emb[:, 1],
        anomaly_emb[:, 2],
        c='red',
        alpha=0.9,
        label='Anomaly',
        zorder=3,  # higher zorder => rendered on top
        marker='^'  # (Optional) use a different marker for clarity
    )

    ax.set_title(f"3D t-SNE Visualization of Linux Bovia PE using {embedding_column}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend()
    plt.show()

def get_bart_embeddings(sentences,model_bart):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_bart(**inputs)
        # outputs.last_hidden_state is (batch_size, seq_len, hidden_dim)
        # A simple approach: take the [CLS]-equivalent (or average pool)
        # Bart doesn't have a classical [CLS], so typically we pool the last hidden layer
        # or take the first token (which is <s>).
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Custom DCG and nDCG functions
def discounted_cumulative_gain(ranks):
    return sum(1.0 / np.log2(rank + 1) for rank in ranks)

def normalized_discounted_cumulative_gain(ranks, num_gt):
    dcg = discounted_cumulative_gain(ranks)
    maxdcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_gt + 1))
    return dcg / maxdcg


# ============
# Step 3: Light VGAE Encoder
# ============
class LightGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logvar = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar


def main(args):


    input=os.path.join(args.source_directory, "ProcessAll.csv")
    groundtruth=args.source_ground_truth
    # Find positions of true positives in the DataFrame
    _processes = pd.read_csv(input)

    _labels_df = pd.read_csv(groundtruth)
    _apt_list = _labels_df.loc[_labels_df["label"] == "AdmSubject::Node"]["uuid"]
    APT_positions = _processes[_processes['Object_ID'].isin(_apt_list)].index.tolist()
    outlier_indices=APT_positions

    pe_df = pd.read_csv(os.path.join(args.source_directory, "ProcessEvent.csv") ) # Binary features: one-hot system events
    px_df = pd.read_csv(os.path.join(args.source_directory, "ProcessExec.csv") ) # One-hot encoded executables
    pp_df = pd.read_csv(os.path.join(args.source_directory, "ProcessParent.csv") ) # Parent → Child binary matrix
    pn_df = pd.read_csv(os.path.join(args.source_directory, "ProcessNetflow.csv") ) # One-hot encoded connections (e.g., ports/IPs)

    df = pd.DataFrame(_processes)
    # Predefined feature name lists
    list_of_events = list(pe_df.columns)[1:]#.remove("Object_ID")
    list_of_executable_names = list(px_df.columns)[1:]
    list_of_netflow_activities = list(pn_df.columns)[1:]
    list_of_spawned = list(pp_df.columns)[1:]


    with open(args.dictionary, "r") as f:
        exec_translation_dict = json.load(f)

    PA=_processes
    ### removing attack items and leaving normal processes for training
    outlier_ids = set(outlier_indices)  # originally indices, but better if you store actual Object_IDs

    PE=pe_df
    PP=pp_df
    PN=pn_df
    PX=px_df
    source_sentences = []
    target_sentences = []

    for idx, row in PA.iterrows():
        obj_id = row["Object_ID"]

        pe_row = PE[PE["Object_ID"] == obj_id]
        px_row = PX[PX["Object_ID"] == obj_id]
        pn_row = PN[PN["Object_ID"] == obj_id]
        pp_row = PP[PP["Object_ID"] == obj_id]

        source_parts = []
        target_parts = []

        if not pe_row.empty:
            active_events = [col for col in pe_row.columns[1:] if pe_row.iloc[0][col] == 1]
            source_parts.extend([f"performed {flow}" for flow in active_events])
            target_parts.extend([col.replace("EVENT_", "performed ") for col in active_events])

        if not px_row.empty:
            active_execs = [col for col in px_row.columns[1:] if px_row.iloc[0][col] == 1]
            source_parts.extend([f"executed {exe}" for exe in active_execs])
            target_parts.extend([f"executed {exec_translation_dict[exe]}" if exe in exec_translation_dict else f"launched {exe}" for exe in active_execs])

        if not pn_row.empty:
            active_netflows = [col for col in pn_row.columns[1:] if pn_row.iloc[0][col] == 1]
            source_parts.extend([f"network activity {flow}" for flow in active_netflows])
            target_parts.extend([f"network activity {flow}" for flow in active_netflows])

        if not pp_row.empty:
            active_spawned = [col for col in pp_row.columns[1:] if pp_row.iloc[0][col] == 1]
            source_parts.extend([f"spawned {flow}" for flow in active_spawned])
            target_parts.extend([f"spawned {exec_translation_dict[exe]}" if exe in exec_translation_dict else f"spawned {exe}" for exe in active_spawned])

        source_sentence = "The process has " + ", ".join(source_parts) + " ."
        target_sentence = "The process has " + ", ".join(target_parts) + " ."

        source_sentences.append(source_sentence)
        target_sentences.append(target_sentence)

    
    sent_df = pd.DataFrame({
        "Object_ID": PA["Object_ID"],
        "Source_Sentence": source_sentences,
        "Translated_Sentence": target_sentences
    })




    #### STEP 6 Generate Embeddings for Normal Pool


    # Load all selected models
    models = {
        "bert": SentenceTransformer("bert-base-nli-mean-tokens"),
    "roberta": SentenceTransformer("roberta-base-nli-stsb-mean-tokens"),
    "distilbert": SentenceTransformer("distilbert-base-nli-stsb-mean-tokens"),
        "minilm": SentenceTransformer("all-MiniLM-L6-v2")
    }

    combined_df=sent_df
    combined_df = combined_df[~combined_df["Object_ID"].isin(_apt_list)]

    sentences = combined_df["Translated_Sentence"].tolist()
    #sentences_translated = combined_df["sentence_translated"].tolist()

    for name, model in models.items():
        print(f"Encoding with {name}...")

        combined_df[f"normal_pool_embedding_{name}"] = model.encode(
            sentences, show_progress_bar=True, convert_to_numpy=True
        ).tolist()

        #combined_df[f"embedding_{name}_translated"] = model.encode(
        #    sentences_translated, show_progress_bar=True, convert_to_numpy=True
        #).tolist()
    combined_df.shape
    embedding_columns = [
        "normal_pool_embedding_bert",
        "normal_pool_embedding_roberta",
    "normal_pool_embedding_distilbert",
        "normal_pool_embedding_minilm"
    ]

    for col in embedding_columns:
        # Get the first embedding (all should have the same shape)
        first_vector = combined_df[col].iloc[0]
        
        if hasattr(first_vector, '__len__'):
            print(f"{col}: Dimension = {len(first_vector)}")
        else:
            print(f"{col}: Not a valid embedding (missing or malformed)")

    

    ###################### TARGET PREPARATION
    #####=================================  Step 2: Get New Unlabeled Target Processes (test set)
    ################=================================
    
    pa_df_target = pd.read_csv(os.path.join(args.target_directory, "ProcessAll.csv")) 
    pe_df_target = pd.read_csv(os.path.join(args.target_directory, "ProcessEvent.csv") ) # Binary features: one-hot system events
    px_df_target = pd.read_csv(os.path.join(args.target_directory, "ProcessExec.csv") ) # One-hot encoded executables
    pp_df_target = pd.read_csv(os.path.join(args.target_directory, "ProcessParent.csv") ) # Parent → Child binary matrix
    pn_df_target = pd.read_csv(os.path.join(args.target_directory, "ProcessNetflow.csv")  )# One-hot encoded connections (e.g., ports/IPs)

    target_groundtruth=args.target_ground_truth

    # Find positions of true positives in the DataFrame
    target_process_ids = pa_df_target["Object_ID"]

    target_labels_df = pd.read_csv(target_groundtruth)
    target_apt_list = target_labels_df.loc[target_labels_df["label"] == "AdmSubject::Node"]["uuid"]
    target_APT_positions = pa_df_target[pa_df_target['Object_ID'].isin(target_apt_list)].index.tolist()
    target_outlier_indices=target_APT_positions

    # Step 0: Initialize combined DataFrame with common Object_IDs
    target_combined_df = pd.DataFrame()
    real_target_sentences = []

    for idx, row in pa_df_target.iterrows():
        obj_id = row["Object_ID"]

        pe_row = pe_df_target[pe_df_target["Object_ID"] == obj_id]
        px_row = px_df_target[px_df_target["Object_ID"] == obj_id]
        pn_row = pp_df_target[pp_df_target["Object_ID"] == obj_id]
        pp_row = pn_df_target[pn_df_target["Object_ID"] == obj_id]

        target_parts = []

        if not pe_row.empty:
            active_events = [col for col in pe_row.columns[1:] if pe_row.iloc[0][col] == 1]
            target_parts.extend([col.replace("EVENT_", "performed ") for col in active_events])

        if not px_row.empty:
            active_execs = [col for col in px_row.columns[1:] if px_row.iloc[0][col] == 1]
            target_parts.extend([f"executed {exe}" for exe in active_execs])

        if not pn_row.empty:
            active_netflows = [col for col in pn_row.columns[1:] if pn_row.iloc[0][col] == 1]
            target_parts.extend([f"network activity {flow}" for flow in active_netflows])

        if not pp_row.empty:
            active_spawned = [col for col in pp_row.columns[1:] if pp_row.iloc[0][col] == 1]
            target_parts.extend([f"spawned {flow}" for flow in active_spawned])

        real_target_sentence = "The process has " + ", ".join(target_parts) + " ."

        real_target_sentences.append(real_target_sentence)

    
    sent_target_df = pd.DataFrame({
        "Object_ID": pa_df_target["Object_ID"],
        "Target_Sentence": real_target_sentences
    })

    target_combined_df=sent_target_df
    #target_combined_df = target_combined_df[~target_combined_df["Object_ID"].isin(target_apt_list)]

    target_sentences = target_combined_df["Target_Sentence"].tolist()

    for name, model in models.items():
        print(f"Encoding with {name}...")

        target_combined_df[f"target_embedding_{name}"] = model.encode(
            target_sentences, show_progress_bar=True, convert_to_numpy=True
        ).tolist()

        #combined_df[f"embedding_{name}_translated"] = model.encode(
        #    sentences_translated, show_progress_bar=True, convert_to_numpy=True
        #).tolist()

    target_combined_df.shape
    target_embedding_columns = [
        "target_embedding_bert",
        "target_embedding_roberta",
    "target_embedding_distilbert",
        "target_embedding_minilm"
    ]

    for col in target_embedding_columns:
        # Get the first embedding (all should have the same shape)
        first_vector = target_combined_df[col].iloc[0]
        
        if hasattr(first_vector, '__len__'):
            print(f"{col}: Dimension = {len(first_vector)}")
        else:
            print(f"{col}: Not a valid embedding (missing or malformed)")





    # ============
    # Step 1: Load distilbert Embeddings
    # ============
    max_source = 1000
    outlier_ids_target = target_apt_list

    source_all = np.stack(combined_df["normal_pool_embedding_distilbert"].values)
    normal_embeddings = source_all#[:max_source]

    target_embeddings = np.vstack(target_combined_df["target_embedding_distilbert"].values)

    print(f"✅ Source shape: {normal_embeddings.shape}, Target shape: {target_embeddings.shape}")

    # ============
    # Step 2: Build Graph from distilbert source embeddings
    # ============
    k = 64
    adj = kneighbors_graph(normal_embeddings, k, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    x = torch.tensor(normal_embeddings, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)


    # ============
    # Step 4: Train VGAE
    # ============
    embedding_dim = 64
    model = VGAE(LightGCNEncoder(x.size(1), embedding_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index) + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

    # ============
    # Step 5: Encode Target distilbert Embeddings
    # ============
    model.eval()
    target_x = torch.tensor(target_embeddings, dtype=torch.float)
    n_target = target_x.shape[0]
    dummy_edge_index = torch.stack([torch.arange(n_target), torch.arange(n_target)], dim=0)

    with torch.no_grad():
        z_source = model.encode(data.x, data.edge_index)

        mu_target = model.encoder.conv_mu(target_x, dummy_edge_index)
        logvar_target = model.encoder.conv_logvar(target_x, dummy_edge_index)
        z_target = model.reparametrize(mu_target, logvar_target)

    # ============
    # Step 6: Compute Cosine-Based Anomaly Score
    # ============
    z_source_np = z_source.cpu().numpy()
    z_target_np = z_target.cpu().numpy()

    sim_matrix = cosine_similarity(z_target_np, z_source_np)
    max_sim = sim_matrix.max(axis=1)
    anomaly_score_sim = 1 - max_sim

    # ============
    # Step 7: Store Scores & Evaluate
    # ============
    target_combined_df["anomaly_score_vgae_distilbert"] = anomaly_score_sim
    target_combined_df["label"] = target_combined_df["Object_ID"].isin(outlier_ids_target).astype(int)

    y_true = target_combined_df["label"].values
    y_scores = target_combined_df["anomaly_score_vgae_distilbert"].values

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    ndcg_builtin = ndcg_score([y_true], [y_scores])

    print(f"AUC (distilbert, soft VGAE): {auc:.4f}")
  





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_directory", type=str, required=True)
    parser.add_argument("--source_ground_truth", type=str, required=True)
    parser.add_argument("--target_directory", type=str, required=True)
    parser.add_argument("--target_ground_truth", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    args = parser.parse_args()
    main(args)