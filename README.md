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
![image](https://github.com/user-attachments/assets/23dc60fd-ce3a-43e2-a097-6dcc290cdb7f)

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

## LSTM Model
The LSTM architecture of the model used is in the image given.
![image](https://github.com/user-attachments/assets/d1770cd4-5261-4dab-ba1c-a1dfd49fee78)

The RNN consists of the following layers:
- Input Layer: The textual data, which has been tokenized. 
- Embedding Layer: Converts input sequences of text into dense vector representations of fixed size (128 dimensions) to capture semantic relationships between words.
- LSTM Layer (64 units): A Long Short-Term Memory layer with 64 units that processes the sequential data, retaining important temporal dependencies between words. return_sequences=True ensures that the output of this layer is passed to the next LSTM layer.
- LSTM Layer (32 units): Another LSTM layer with 32 units to capture further sequential dependencies in the data.
- Dense Layer (64 units): A fully connected layer with 64 units and ReLU activation, which adds non-linearity and helps the model learn complex patterns.
- Dropout Layer (0.5 rate): A dropout layer with a 50% dropout rate to reduce overfitting by randomly setting a fraction of input units to 0 during training.
- Output Layer (3 units): A dense output layer with 3 units (corresponding to the Safe, Neutral, and Harmful categories), and a softmax activation function to produce probability distributions over the output classes.

## LSTM Model Training
- Loss Function: Since this is a multi-class classification problem, we used categorical cross-entropy as the loss function. This is appropriate for problems where each input can belong to one of several classes.
- Optimizer: The Adam optimizer was used for its ability to adaptively adjust the learning rate during training. Adam is a popular choice for training deep learning models because it generally converges faster and provides good results.
- Model Training Parameters:
- Epochs: 10 epochs were used for training, meaning the model passed through the entire training dataset 10 times.
- Batch Size: A batch size of 32 was used, which determines how many samples are processed before updating the model's weights.
- Validation Split: A portion of the training data (20%) was used for validation, allowing the model to tune its parameters and avoid overfitting.

## LSTM Model Evaluation
### Model Evaluation
#### Performance Metrics
The performance of the LSTM model was evaluated using precision, recall, F1-score, and accuracy. The classification report for the model on the test set is as follows:
#### Explanation
- Precision: The model's ability to correctly classify positive instances for each class (Safe, Neutral, Harmful). High precision indicates that when the model predicts a class, it is likely correct.
- Recall: The model's ability to capture all the true instances of each class. A high recall for a class indicates that the model is effective at identifying that class.
- F1-Score: The harmonic mean of precision and recall, providing a balance between the two. A higher F1-score signifies a better overall performance, especially in the presence of imbalanced data.
- Accuracy: The proportion of correctly classified instances across all classes. The model achieved an accuracy of 85% on the test set.
### Confusion Matrix
The LSTM model's performance was assessed using key metrics such as precision, recall, F1-score, and accuracy. Below is the classification report summarizing the results:
![image](https://github.com/user-attachments/assets/ad24d076-56a6-48e0-bbed-51b47853f0b2)
 
### Explanation
1. Precision: Measures the accuracy of the model when predicting a particular class (Safe, Neutral, Harmful). Higher precision for "Harmful" indicates fewer false positives.
2. Recall: Indicates the model's ability to detect all instances of a class. A higher recall for "Neutral" means the model can successfully capture more neutral content.
3. F1-Score: A balanced measure combining precision and recall. The overall F1-scores are decent for each class, showing a good trade-off between precision and recall
4. Accuracy: Overall, the model achieved an accuracy of 85%, meaning 85% of the predictions on the test set were correct.

![image](https://github.com/user-attachments/assets/3c5b7323-4bc8-4261-a3f8-fbc1a2de9bfc)

# Improvements
The UI (flask) component needs to be developed. In production we can have a REST API built for the Youtube Content Moderation module.
The client can POST the youtube URL via REST API
The server runs the inference module using the URL and obtains the response
The response is relayed back via REST API response to the client
We can reduce overfitting and improve the accuracy of the prediction by tuning the hyperparameters.

	



