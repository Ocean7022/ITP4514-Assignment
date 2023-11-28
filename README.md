
![img](img/Cove%20page%20img.png)

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
    "link": "https://www.cnn.com/politics/live-news/republican-debate-11-08-23/index.html",
    "title": "Third 2024 Republican presidential debate",
    "type": "politics",
    "content": "The third Republican presidential..."
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

# Use Model
### Classificatino
|Model|Done At|
|-|-|
|Naive Bayes (NB)|23-11-2023|
|Recurrent Neural Network (RNN)|Not done yet|

### Summerization
|Model|Done At|
|-|-|
|Bidirectional Encoder Representations from Transformers (BERT)|23-11-2023|
|T5|25-11-2023|

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

# Parameter Tuning in NB
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

# Tuning Detail in NB
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


# Work load
|Action|Done At|Done By|Source|Remark|
|-|-|-|-|-|
|Create NB Model Evaluation|10-11-2023|Ocean|src/nbModelEvaluation.py||
|Create Cawler|15-11-2023|Ocean|dataCawler/nbcnews.py|https://www.nbcnews.com/us-news|
|Create Cawler|17-11-2023|Roy|dataCawler/theStandard.py|https://www.thestandard.com.hk/|
|Create Cawler|17-11-2023|David|dataCawler/aljazeeranews.py|https://www.aljazeera.com/|
|Create NB Feature Analysis|18-11-2023|Ocean, David|src/nbFeatureAnalysis.py||
|Add Stop Wrors and Tuning|23-11-2023|Ocean, Roy , David|data/stopWordList.csv||
|Model Tuning|23-11-2023|Ocean, Roy , David|src/config/Config.py||
||||||
||||||
||||||
||||||


# Installition
#### Please install python 3.11.6 in virtual environment
```bash
pip install -r requirements.txt
```