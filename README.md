# Sentiment Analysis with Emotion Intensity Prediction

## Project Overview
This project focuses on building machine learning models to predict the sentiment of tweets by quantifying the intensity of specific emotions expressed. The sentiment analysis assigns a score between 0 and 1 to indicate the strength of a particular emotion, such as joy, sadness, fear, or anger, conveyed in the tweet. A higher score signifies a stronger presence of the emotion, while a lower score indicates its absence or minimal impact.

## Dataset
- **Source:** [Emotion Intensity Shared Task](https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)
- **Format:** Text files
- **Training and Test Data:** Separate datasets are provided for each of the four emotions (joy, sadness, fear, and anger). The training datasets include tweets along with real-valued scores representing the intensity of the corresponding emotion felt by the speaker. The test data consists of tweet text only.

## Tools and Packages
- **Data Management:** Pandas
- **Numerical Computations:** NumPy
- **Visualization:** Matplotlib, Seaborn
- **Text Processing:** NLTK
- **Machine Learning:** Scikit-learn (for linear regression and regularization), TensorFlow, Keras (for neural network)
- **Data Preprocessing:** StandardScaler

## Models Developed
### 1) Linear Regression Models
- **Linear Regression:** Utilized to predict emotion intensity scores.
- **Regularized Linear Regression:** Employed L2 regularization to mitigate overfitting.

### 2) Multilayer Neural Network
- **Model Architecture:** Multilayer neural network with dense layers.
- **Optimization:** Adam optimizer utilized with Mean Squared Error (MSE) loss function.
- **Performance Metric:** Mean Absolute Error (MAE) used to evaluate model performance.

## Usage
1. **Data Preprocessing:** Text data was preprocessed using NLTK for tasks such as tokenization and stop word removal.
2. **Model Training:** Separate models were trained for each emotion using the provided training datasets.
3. **Evaluation:** Model performance was evaluated using the test datasets, and metrics such as MAE and RMSE were computed.
4. **Comparison:** The performance of different models and approaches was compared to select the most effective one for each emotion.

## Repository Structure
- **Data:** Contains training and test datasets for each emotion.
- **Notebooks:** Jupyter notebooks containing code for data preprocessing, model training, and evaluation.
- **Models:** Saved models and model performance metrics.
- **Reports:** Project reports, analysis summaries, and visualizations.
- **README.md:** Main repository file providing an overview of the project, instructions, and usage guidelines.



## Contributors
- Pratheek


