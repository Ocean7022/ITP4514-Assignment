
![img](img/Cove%20page%20img.png)

---
---
- [Data Collection Steps](#data-collection-steps)
  - [News Links Format](#news-links-format)
  - [News Data Format](#news-data-format)
- [News Cawler Time](#news-cawler-time)
  - [Aljazeera News](#aljazeera-news)
  - [CCN News](#ccn-news)
  - [The Standard](#the-standard)
  - [NBC News](#nbc-news)
  - [Total News](#total-news)
- [Use Model](#use-model)
  - [Classificatino](#classificatino)
  - [Summerization](#summerization)
- [Naive Bayes classifier (NB)](#naive-bayes-classifier-nb)
  - [Parameter Tuning in NB](#parameter-tuning-in-nb)
  - [Tuning Detail in NB](#tuning-detail-in-nb)
    - [Performance Summary](#performance-summary)
    - [Class Imbalance](#class-imbalance)
    - [TfidfVectorizer()](#tfidfvectorizer)
- [Tuning Detail in GRU](#tuning-detail-in-gru)
  - [Data Preprocessing](#data-preprocessing)
  - [Reasons for Using Class Weights](#reasons-for-using-class-weights)
  - [Tuning Setting](#tuning-setting)
  - [Step-by-Step Breakdown](#step-by-step-breakdown)
  - [Early Stopping](#early-stopping)
---
---

# Data Collection Steps
```mermaid
graph TD
cawler01[The Standard]
cawler02[Aljazeera News]
cawler03[NBC News]
cawler04[CCN News]
getLinks01[Get Links]
getLinks02[Get Links]
getLinks03[Get Links]
getLinks04[Get Links]
getData01[Get Data]
getData02[Get Data]
getData03[Get Data]
getData04[Get Data]
composeData[Compose Data]
dataSet[(Data Set)]

cawler01 --> getLinks01 --> getData01 --> composeData
cawler02 --> getLinks02 --> getData02 --> composeData
cawler03 --> getLinks03 --> getData03 --> composeData
cawler04 --> getLinks04 --> getData04 --> composeData
composeData -- Clean Data --> dataSet
```

#### News Links Format
```json
{
    "link": "https://www.cnn.com/politics/live-news/republican-debate-11-08-23/index.html",
    "title": "Third 2024 Republican presidential debate",
    "type": "politics"
}
```
#### News Data Format
```json
{
    "title": "Third 2024 Republican presidential debate",
    "content": "The third Republican presidential...",
    "category": "politics",
    "link": "https://www.cnn.com/politics/live-news/republican-debate-11-08-23/index.html"
}

```
## News Cawler Time
#### Aljazeera News
![img](img/AljazeeraCawler.png)
#### CCN News
![img](img/CCNCawler.png)
#### The Standard
![img](img/theStandardCawler.png)
#### NBC News
![img](img/NCBCawler.png)

## Total News
#### We clean data in this step
![img](img/TotalNews.png)

---
---

# Use Model
### Classificatino
|Model|Done At|
|-|-|
|Naive Bayes (NB)|23-11-2023|
|Gated Recurrent Units (GRU)|27-11-2023|

### Summerization
|Model|Done At|
|-|-|
|Bidirectional Encoder Representations from Transformers (BERT)|23-11-2023|
|T5|25-11-2023|

---
---

# Naive Bayes classifier (NB)
```mermaid
graph TD
step01[Data Collection]
step02[Clean Data]
step03[Model Training]
step04[Model Evaluation]
if01{Good Model?}
step05[Parameter Tuning]
step06[Model Deployment]

step01 --> step02 --> step03 --> step04 --> if01
if01 -- No --> step05 --> step03
if01 -- Yes --> step06
```

## Parameter Tuning in NB
```mermaid
graph TD
step00[Start]
step01[Check Result]
if01{Good Result?}
step02[Start Tuning]
step03[Model Deployment]
subgraph Tuning
    step04[Check top 10 words in each category]
    step05[Add words to stop word list]
    step06[Set vectorize parameter]
end

step00 -- NB-ModelEvaluation.py --> step01 --> if01
if01 -- Yes --> step03
if01 -- NO --> step02
step02 -- NB-FeatureAnalysis.py --> step04
step04 --> step06 -- NB-ModelEvaluation.py --> step01
step04 --> step05 -- NB-ModelEvaluation.py --> step01
```

## Tuning Detail in NB
|Before Tuning|After Tuning|
|-|-|
|Vectorize![img](img/NB-Tuning/Vectorize-Befor.png)|Vectorize![img](img/NB-Tuning/Vectorize-After.png)|
|NB-ModelEvaluation.py![img](img/NB-Tuning/01-Befor.png)|NB-ModelEvaluation.py![img](img/NB-Tuning/01-After.png)|
|NB-FeatureAnalysis.py![img](img/NB-Tuning/01-Befor2.png)|NB-FeatureAnalysis.py![img](img/NB-Tuning/01-After2.png)|

### Performance Summary
- The categories **business** and **sport** showed good predictive performance with high precision and recall, resulting in relatively high f1-scores.
- The **health** category, despite having a precision of 1.00, had a very low recall of 0.06, indicating that the model hardly identified any true positives in this category.
- The categories **culture**, **education**, **politics**, **property**, **technology**, and **travel** all had precision and recall of 0, indicating that the model failed to correctly identify any samples from these categories.
### Class Imbalance
- The **support** values indicate the number of samples for each category. There is a clear class imbalance in the dataset, with the **business** (3581) and **sport** (3175) categories having significantly more samples than other categories.

### TfidfVectorizer()
| Setting| Value|Explanation|
|-|-|-|
|`max_features`|8000|Specifies the maximum number of features to consider. Only considers the top 8000 terms by term frequency. Helps in limiting the size of the model and computational complexity.|
|`ngram_range`|(1, 3)| Defines the range of n-grams to be considered. Here, (1, 3) means that unigrams, bigrams, and trigrams will be used. This expands the feature set but also increases computational load.|
|`stop_words`|List of stop words from `ENGLISH_STOP_WORDS` union with custom list from `stopWordListPath` CSV|Removes common stop words to reduce noise. These are typically words that don't carry significant meaning (like "and", "the", etc.). Here, it uses sklearn's English stop words combined with a custom list from a CSV file.|
|`token_pattern`|`\b[a-zA-Z]{2,}\b`|Defines the regex pattern for tokens (like words) to be considered. This pattern means only words with at least two letters are considered. Helps in excluding single-letter words, possibly typos or meaningless characters.|
|`max_df`|0.5|Sets the maximum document frequency for terms. If a term appears in more than 50% of the documents, it will be excluded. Helps in excluding too common terms which might not be helpful for classification or clustering.|
|`min_df`|3|Sets the minimum document frequency for terms. A term must appear in at least 3 documents to be included. Helps in excluding rare terms which might not contribute to the analysis of most documents.|
|`norm`|`l2`|Specifies the normalization method. `l2` normalization ensures all feature vectors have a Euclidean length of 1. Helps in mitigating the effect of document length on weights.|
|`sublinear_tf`|True|Enables sublinear frequency scaling. Converts term frequency to 1 + log(tf), which helps in reducing the impact of high-frequency terms.|

---
---

# Tuning Detail in GRU
### Data Preprocessing
![img](img/GRU-Tuning/DataPreprocessing.png)

---

### Reasons for Using Class Weights
#### Due to the highly uneven distribution of data we have collected, we must use class weights. It can help mitigate the bias towards the more common classes, thereby improving the model's performance in predicting minority class instances.
![img](img/GRU-Tuning/typeOfNewspng.png)
### 1. Class Imbalance
- **Issue**: When certain classes are significantly more represented than others in a dataset, models might develop a bias towards predicting these majority classes, at the expense of minority ones.
- **Solution**: Class weights help the model to pay more attention to the less represented classes, thereby improving overall performance in scenarios with imbalanced data.

### 2. Enhancing Recognition of Minority Classes
- **Importance**: In some applications, correctly identifying minority classes (e.g., rare diseases in medical diagnosis) is more crucial than identifying majority classes.
- **Approach**: Class weights can be adjusted to increase the importance of minority classes, enhancing the model's accuracy and recognition rate for these classes.

### 3. Preventing Overfitting
- **Challenge**: Training on imbalanced datasets can lead to overfitting to the majority class.
- **Strategy**: By adjusting class weights, models can be encouraged to learn features more comprehensively across all classes, rather than focusing predominantly on the most common classes.

### 4. Improving Model Generalization
- **Goal**: A model that performs well across various scenarios, not just on the majority class.
- **Method**: Class weights encourage consideration of all classes during training, improving the model's ability to generalize and perform robustly in diverse situations.

### 5. Meeting Specific Business or Research Objectives
- **Customization**: Researchers or business decision-makers might have specific needs where the model should focus more on certain classes, regardless of their prevalence in the dataset.
- **Custom Approach**: Class weights allow for the customization of training objectives, ensuring that the model pays more attention to specific classes as required.

---

### Tuning Setting
#### Since both the training and testing sets are randomly generated.
#### Each configuration will be run 5 times, and the best result will be recorded.
#### During the process, the `batch_size` will be adjusted appropriately to prevent excessive GPU memory consumption, which can lead to a slowdown in training speed.
|Step|max_length|vocab_size|embedding_dim|hidden_size|num_layers|learning_rate|num_epochs|Early stopping|class_weights|Accuracy|
|-|-|-|-|-|-|-|-|-|-|-|
|01|1500|Full Size(161019)|300|128|3|0.001|10|No|No|90.80%|
|02|1500|Full Size(161019)|`600`|128|3|0.001|10|No|No|91.31%|
|03|1500|Full Size(161019)|600|128|`6`|0.001|10|No|No|92.57%|
|04|1500|Full Size(161019)|600|`256`|6|0.001|10|No|No|92.62%|
|05|1500|Full Size(161019)|`400`|128|6|0.001|`20`|`Yes(12/20)`|`Yes`|93.53%|
|06|1500|`14000`|400|128|6|0.001|20|`Yes(11/20)`|Yes|94.37%|
|07|1500|14000|400|128|`3`|0.001|20|Yes(11/20)|Yes|94.89%|
|08|`1200`|14000|400|128|3|0.001|20|`Yes(9/20)`|Yes|94.49%|
|09|`1000`|14000|400|128|3|0.003|20|Yes(9/20)|Yes|95.41%|

---

### Step-by-Step Breakdown
#### Step 01 to 02
- **Change**: Increased `embedding_dim` from 300 to `600`.
- **Impact**: 
  - **Advantage**: Enhanced word feature representation.
  - **Disadvantage**: Higher model complexity, longer training.
- **Outcome**: Accuracy improved from 90.80% to 91.31%.

#### Step 02 to 03
- **Change**: Increased `num_layers` from 3 to `6`.
- **Impact**:
  - **Advantage**: Better pattern recognition.
  - **Disadvantage**: Increased risk of overfitting, longer training.
- **Outcome**: Accuracy up to 92.57%.

#### Step 03 to 04
- **Change**: Increased `hidden_size` to `256`.
- **Impact**:
  - **Advantage**: More complex data processing.
  - **Disadvantage**: Higher computational demand.
- **Outcome**: Slight increase in accuracy to 92.62%.

#### Step 04 to 05
- **Change**: Reduced `embedding_dim` to `400`, increased `num_epochs` to `20`, introduced `Early stopping` and `class_weights`.
- **Impact**:
  - **Advantage**: Improved generalization, optimal stopping.
  - **Disadvantage**: Possible early halt before best model.
- **Outcome**: Significant rise in accuracy to 93.53%.

#### Step 05 to 06
- **Change**: Reduced `vocab_size` to `14000`.
- **Impact**:
  - **Advantage**: More focus, less noise.
  - **Disadvantage**: Loss of rare words.
- **Outcome**: Increased accuracy to 94.37%.

#### Step 06 to 07
- **Change**: Reduced `num_layers` to `3`.
- **Impact**:
  - **Advantage**: Faster training, reduced overfitting.
  - **Disadvantage**: Less capacity for very complex patterns.
- **Outcome**: Peak accuracy of 94.89%.

#### Step 07 to 08
- **Change**: Decreased `max_length` to `1200`.
- **Impact**:
  - **Advantage**: Quicker processing, potential noise reduction.
  - **Disadvantage**: Potential loss of context.
- **Outcome**: Slight drop in accuracy to 94.49%.

#### Step 08 to 09
- **Change**: Further reduced `max_length` to `1000`, increased `learning_rate` to `0.003`.
- **Impact**:
  - **Advantage**: Faster convergence, focus on core content.
  - **Disadvantage**: Risk of context loss, overshooting minimum loss.
- **Outcome**: Highest accuracy at 95.41%, early stop at 9/20 epochs.

#### General Observations
- Balancing model complexity and computational efficiency was key.
- Early stopping and class weights significantly enhanced accuracy and model robustness.
- Continuous parameter tuning optimized model performance.

---
### Early Stopping
#### Early stopping is a form of regularization used to avoid overfitting when training a machine learning model, particularly in neural networks. Let's break down what each part of the code does:
```python
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    patience_counter = 0
    torch.save(model.state_dict(), config.GRU_classificationModelPath)
```
- This block checks if the average validation loss (`avg_val_loss`) for the current epoch is less than the best (lowest) validation loss seen in previous epochs (`best_val_loss`).
- If `avg_val_loss` is lower, it means the model has improved. Hence, `best_val_loss` is updated with the new lower value.
- The `patience_counter` is reset to 0, as the model has improved in this epoch.
- The current state of the model is saved. This is typically done to ensure that you have the model parameters that gave the best performance on the validation set.
```python
else:
    patience_counter += 1
    if patience_counter >= patience:
        print('Early stopping triggered')
        break
```
- If `avg_val_loss` is not lower than `best_val_loss`, it implies that the model has not improved in this epoch. The `patience_counter` is incremented by 1.
- If the `patience_counter` reaches a predefined threshold (`patience`), it triggers the early stopping mechanism. This threshold is the number of epochs you're willing to wait for an improvement in model performance.
- When early stopping is triggered, the training loop breaks, stopping further training. This is done to prevent overfitting.
#### In summary, this code saves the best model and stops training if the model does not improve for a specified number of epochs (`patience`). It's an effective way to monitor and halt the training process at the right time, thus safeguarding against overfitting while ensuring the best possible performance on unseen data.

---
---

# Work load
|Action|Done At|Done By|Source|Remark|
|-|-|-|-|-|
|Create NB Model Evaluation|10-11-2023|Ocean|src/nbModelEvaluation.py||
|Create Cawler|15-11-2023|Ocean|dataCawler/nbcnews.py|https://www.nbcnews.com/us-news|
|Create Cawler|17-11-2023|Roy|dataCawler/theStandard.py|https://www.thestandard.com.hk/|
|Create Cawler|17-11-2023|David|dataCawler/aljazeeranews.py|https://www.aljazeera.com/|
|Create Cawler|19-11-2023|Ocean|dataCawler/cnnnews.py|https://www.cnn.com/|
|Create NB Feature Analysis|18-11-2023|Ocean, David|src/nbFeatureAnalysis.py||
|Add Stop Wrors and Tuning|23-11-2023|Ocean, Roy , David|data/stopWordList.csv||
|NB Model Tuning|23-11-2023|Ocean, Roy , David|||
|Create and Tuning GRU Model|27-11-2023|Ocean|src/RNN-ModelTest.py||
||||||
||||||
||||||
||||||

---
---

# Installition
#### Please install library in virtual environment - python 3.11.6
```bash
pip install -r requirements.txt
```