# 🎵 Music Emotion Predictor: Unlocking Emotions in Music

# ✨ Project Overview
Welcome to Music Emotion Predictor! This project delves into the fascinating world of music emotion recognition, leveraging machine learning to predict the emotional "vibe" of a song based on its audio features. Ever wondered if a track is more "Energetic" or "Calm"? This model aims to provide insights into the emotional landscape embedded within musical compositions.

We explore both supervised and unsupervised learning techniques to classify music into four distinct emotional categories: Calm, Sad, Energetic, and Happy.

# 🚀 Key Features
Emotion Prediction: Classifies music into 4 core emotional states.

Multiple ML Models: Implements and compares:

Decision Tree Classifier

K-Nearest Neighbors (KNN) Classifier

K-Means Clustering (for unsupervised grouping)

Comprehensive Data Preprocessing: Includes robust feature scaling and data splitting.

Interactive Visualizations: Utilizes matplotlib, seaborn, and plotly for insightful data exploration and result representation.

Modular Codebase: Organized within a Jupyter Notebook for easy understanding and execution.

# 💡 How It Works
Data Acquisition: The project starts by loading a rich dataset containing various audio features and pre-labeled emotional categories for a vast collection of music tracks.

Feature Engineering & Preprocessing: Raw audio features (like danceability, energy, valence, tempo, etc.) are prepared for model training. This involves cleaning, feature selection, and scaling to ensure optimal model performance.

Model Training: Supervised learning models (Decision Tree, KNN) are trained on the labeled data, learning the patterns that distinguish different emotions. An unsupervised model (K-Means) groups similar songs based on their features.

Emotion Prediction & Evaluation: The trained models then predict emotions on unseen music data. Their performance is evaluated using appropriate metrics, and the distribution of predicted moods is visualized.

# 🛠️ Technologies Used
This project is built using the following powerful tools and libraries:

Python

Jupyter Notebook

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Plotly

# 📊 Dataset
The core of this project relies on the 278k_labelled_uri.csv dataset. This comprehensive dataset provides a rich collection of audio features (e.g., duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, spec_rate) coupled with pre-assigned emotional labels.

Source: https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset 
