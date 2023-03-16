# kaggle Solution of learning-equality-curriculum-recommendations
https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/overview   
First of all I would like to Thanks to Kaggle and The Learning Agency Lab, and thanks to kagglers.

I worked on kaggle for the first time.
The result was 247th place out of 1057 teams. I tried my best as a beginner, but I couldn't get a score higher than the benchmark.

This time, keeping a record of what I have been doing for three weeks.

## Competition content
The goal of this competition is to streamline the process of matching educational content to specific topics in a curriculum.  
In other words,  We select several content candidates for each topic.  
The evaluation method is calculated by the F2 score, and the score increases by giving as few incorrect answers as possible.

## Summary
We will use two models. First, we use the **Retriever model** to extract the top50 candidates by KNN.
Then the **Reranking model** sorts the candidates to narrow down to the more accurate candidates.

**Retriever model :** sentence-transformers/all-MiniLM-L6-v2

**Reranking model :** Trained model from Retriever model

There are 4 steps to create the final submission file.

## CV Strategy
Using KFold, we divided the train data into four. We split fold 0 as evaluation data and folds 1-3 as training data.

## preprocess
For the topic and content sentences, the token preprocessing was done as follows.

**topics :** title + [SEP] + description + [SEP] + breadcrumbs(topic-tree)

**content :**  title + [SEP] + description + [SEP] + text

All the titles and descriptions are included in the features, and the breadcrumbs and text are truncated according to the limit.

## Step1: Retriever
We pretrained models from sentence-transformer library, and we used MultipleNegativesRankingLoss. It took 6 hours to train the model.

**Model :** sentence-transformers/all-MiniLM-L6-v2  
**Sequence Length :** 64  
**Epochs :** 30  
**Batch Size :** 288  
**Loss function :** MultipleNegativesRankingLoss  

Recall@50 score resulted in:

- 0.870 for whole data  
- 0.814 for valid data

## Step2: Clustering
Calculate a vector for each content and use KNN to create the top 50 candidates.  
Using embeddings and select candidate @50 for each topic with close vector distance, save to csv file.

## Step3: Reranking 
Using the step1 model, we train the model so that the F2 score increases while changing the order of the candidates.
It took 30 hours.  
Probably because this part was wrong, it seems that the score did not improve.  
Later I will check other people's solutions again and study better methods.

**Base Model :** Trained model from Stage 1    
**Sequence Length :** 128  
**Epochs :** 10  
**Batch Size :** 384  
**Loss function :** BCEWithLogitsLoss  

The F2 score is  
- 0.476 CV

## Step4: Inference
Use the threshold got when reranking in step3 to make inferences on the test data.  
There is a difference of about 0.3 between CV and LB.
However, since everyone's scores increased by 0.3 to 0.4, it means that the remaining 70% of the test data was predicted well.

The F2 score is  
- 0.442 Public LB
- 0.465 Private LB

## Not worked
- Other pretrained model (e.g. sentence-transformers/all-MiniLM-L12-v2)
- Change the number of KNN candidates (e.g. @10, @100)
