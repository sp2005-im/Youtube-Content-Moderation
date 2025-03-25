# Youtube-Content-Moderation
YouTube content moderation is essential to ensure user safety, comply with legal regulations, prevent misinformation, maintain platform integrity, and protect advertisers by leveraging AI and human oversight to detect and remove harmful content. 
We collected metadata from 600 YouTube videos and conducted sentiment analysis using a Random Forest classifier, achieving 92% accuracy. Additionally, we developed an NLP classifier based on Long Short-Term Memory (LSTM) networks to categorize metadata as safe, neutral, or harmful, attaining an accuracy of 85%.
# Introduction
YouTube, as one of the largest content-sharing platforms, hosts vast amounts of user-generated videos, making content moderation a critical challenge. While manual moderation is effective, it is labor-intensive and does not scale efficiently with the platform’s rapid content growth. Automated approaches leveraging machine learning and natural language processing (NLP) offer a viable solution for identifying harmful or inappropriate content based on video metadata.
In this study, we collected metadata from 600 YouTube videos and applied two machine learning models for classification: a Random Forest (RF) classifier for sentiment analysis and a Long Short-Term Memory (LSTM) network for NLP-based classification. The RF classifier is an ensemble learning method that operates by constructing multiple decision trees and aggregating their outputs, making it robust against overfitting and effective for structured data analysis. On the other hand, LSTM, a specialized type of recurrent neural network (RNN), is well-suited for sequential data processing, allowing it to capture contextual dependencies in textual metadata.
Our approach achieved 92% accuracy with the RF classifier for sentiment-based classification and 85% accuracy with the LSTM model for identifying whether video metadata is safe, neutral, or harmful. These results highlight the potential of machine learning in improving YouTube content moderation, providing scalable and efficient solutions for identifying harmful content.
# Methodology for Sentiment Analysis
## Data Collection
To perform sentiment analysis on YouTube metadata we collected metadata from 600 Youtube videos across different content categories, classified as Safe, Neutral and Harmful. The data collection process involved:
1. Defining 12 categories grouped into three sentiment-based labels:
   - Safe Content (e.g., education, music, sports, travel)
   - Neutral Content (e.g., news, discussions, technology, finance)
   - Harmful Content (e.g., violence, explicit content, hate speech, accidents)
3. Using the YouTube Data API (Youtube Data API v3) to search for videos based on predefined keywords.
4. Extracting relevant metadata, including title, description, channel name, publish date, view count, and like count.

## Preprocessing of metadata
Before performing sentiment analysis, we processed the text metadata (titles and descriptions) through several NLP techniques:
1. Cleaning: Removing HTML tags, URLs, punctuation, and extra spaces.
2. Tokenization: Splitting text into individual words.
3. Stopword Removal: Filtering out common words that do not contribute to sentiment analysis.
4. Lemmatization: Converting words to their base forms (e.g., "running" → "run").
The cleaned and processed text data was then combined into a single text feature (combined_text) for sentiment classification.

## Feature Extraction with TF-IDF
To convert textual data into a machine-readable format, we used Term Frequency-Inverse Document Frequency (TF-IDF) vectorization:
1. Extracted the top 5000 features (words) based on their importance.
2. Represented each video’s metadata as a numerical vector for machine learning classification

## Sentiment Classification using Random Forest
- For sentiment analysis, we trained a Random Forest Classifier:
- Input Features: TF-IDF transformed text (combined_text).
- Output Labels:
  - 0 → Safe
  - 1 → Neutral
  - 2 → Harmful
- Model Parameters:
- n_estimators=100 (100 decision trees for ensemble learning).
- class_weight='balanced' to handle class imbalances.
- random_state=42 for reproducibility.

## Evaluation and Results
### Performance metrics
To assess the effectiveness of our classification model, we evaluated it using accuracy, precision, recall, and F1-score. The classification report for the Random Forest classifier is as follows:
| Class       | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Safe (0)   | 1.00      | 0.88   | 0.94     | 43      |
| Neutral (1)| 1.00      | 0.86   | 0.93     | 36      |
| Harmful (2)| 0.80      | 1.00   | 0.89     | 41      |
| **Accuracy** | **-**   | **-**  | **0.92** | **120** |
| Macro-Avg  | 0.93      | 0.91   | 0.92     | 120     |
| Weighted-Avg| 0.93     | 0.92   | 0.92     | 120     |

### Confusion Matrix
### Evaluation Summary

The classification report provides a detailed performance evaluation of the Random Forest model used for YouTube metadata classification into Safe, Neutral, and Harmful categories. The key observations are:
- High Overall Accuracy: The model achieved an accuracy of 92%, indicating strong performance in classifying videos correctly.
- Class-wise Performance:
  - Safe Content (Precision: 1.00, Recall: 0.88, F1-Score: 0.94) – The model predicted safe content well but occasionally misclassified some safe videos.
  - Neutral Content (Precision: 1.00, Recall: 0.86, F1-Score: 0.93) – Slight misclassification occurs between Neutral and Harmful categories.
  - Harmful Content (Precision: 0.80, Recall: 1.00, F1-Score: 0.89) – The model correctly identifies all harmful content (Recall = 1.00) but has a slightly lower precision, meaning some non-harmful videos may be classified as harmful.
- Confusion Matrix Insights:
  1. The confusion matrix shows misclassification primarily between Safe and Harmful categories.
  2. Five Safe videos were classified as Harmful, and 5 Neutral videos were classified as Harmful, suggesting some overlap in content patterns.
 
# Methodology and Analysis for NLP Classifier
To classify YouTube metadata into Safe, Neutral, and Harmful categories, we implemented an LSTM-based deep learning model. The development process included the following steps:

## Data Pre-processing
- The same dataframe that was made from the preprocessing of data used for random forest classification was used as the dataframe for the NLP classifier.
- The dataset was split into features (X = df['combined_text']) and labels (y = df['category_label']).
- Labels were one-hot encoded using to_categorical().
- Data was split into training and testing sets using an 80-20 split.

## Text Tokenization and Sequence Padding 
- A Tokenizer was used with a vocabulary size of 10,000 words.
- The text data was converted into sequences and padded to a fixed length of 200 for uniform input size.


