

## Naive Bayes classifier
```mermaid
graph TD
step01[Data Collection]
step02[Clean data]
step03[Model Training]
step04[Model Evaluation]
if01{Good Model?}
step05[Parameter Tuning]
step06[Model Deployment]

step01 --> step02 --> step03 --> step04 --> if01
if01 -- No --> step05 --> step03
if01 -- Yes --> step06

```

### 1.