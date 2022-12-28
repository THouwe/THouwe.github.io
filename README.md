# [Project 1: Decision Tree Binary Classification (Python)](https://github.com/THouwe/classification_decisionTrees_shippingData_Python)

This [Jupyter Notebook](https://github.com/THouwe/classification_decisionTrees_shippingData_Python/blob/main/shippingData_decisionTree.ipynb) shows how to perform model selection for decision trees in binary classification problems. Specifically, it focuses on deciding on whether to (i) over-/under-sample the data in case of imbalanced datasets, (ii) use gini or entropy as information gain criterion, and (iii) limiting tree depth. It uses this [dataset](https://www.kaggle.com/datasets/prachi13/customer-analytics).

It is divided in the following sections:

- A. Decision Tree (DT): Summary
- B. Data inspection & preparation
- C. Model selection
- D. Model predictions & interpretation

<img src="/images/model_comparison.png" width="500" height="375">



# [Project 2: Sentiment Analysis (Python)](https://github.com/THouwe/NLP_sentimentAnalysis_BTC_Python)

This [Jupyter Notebook](https://github.com/THouwe/NLP_sentimentAnalysis_BTC_Python/blob/main/BTC_sentiment_analysis.ipynb), shows how to download and preprocess 100.000 Bitcoin (BTC) tweets from 2022-11-01 to 2022-11-15 to perform sentiment analysis to gather market intelligence. The aim of the analysis is to understand what are people’s opinions about Bitcoin tweets (i.e., to extract subjectivity and sentiment polarity from text data).

Web scraping is done using [snscrape](https://github.com/JustAnotherArchivist/snscrape). \
Filtering for English language is done using [langdetect](https://github.com/Mimino666/langdetect). \
Sentiment analysis is done using [spacytextblob](https://github.com/SamEdwardes/spacytextblob).

It is divided in the following sections:

- A. Web scraping
- B. Text preprocessing
- C. Sentiment analysis

<img src="/images/sentiment3.JPG" width="475" height="350">



# [Project 3: Time-Series Forecast (R)](https://github.com/THouwe/timeSeriesForecasting_waterInflow_R)
This [R Markdown](https://github.com/THouwe/timeSeriesForecasting_waterInflow_R/blob/main/WaterFlow_Sector229.Rmd) shows how to pre-process and analyze time-series (TS) data. The file shows how to explore seasonalities (e.g., autocorrelation) and trends (e.g., moving averages), and how to perform automated model selection using the [Fable library](https://fable.tidyverts.org/). A description in Italian is provided in [this html file](https://github.com/THouwe/timeSeriesForecasting_waterInflow_R/blob/main/WaterFlow_Sector229.html).

<img src="/images/sector255_predictions.JPG" width="500" height="400">



# [Project 4: Online Speech Data Collection (JavaScript)](https://github.com/THouwe/timeSeriesForecasting_waterInflow_R)
This repo contains the code for a full online speech recognition experiment using JavaScript and the [JSpsych](https://www.jspsych.org/7.3/) library. Speech and text data recorded from the user's browser and are sent to Dropbox's API. The experiment was originally deployed in [Heroku](https://articulatorycontroldemo.herokuapp.com/). However, link is broken due to changes in Heroku's policies. Will be hosted elsewhere soon.


