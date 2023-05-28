
# Fold, Normalize, and Reduce (FNR) - Normalization of Linkedin Datasets with Long Short-Term Memory on Folded Bipartite Graph Network Embedding


This repository includes the implementaton of Long Short-Term Memory Including the Normalization of Linkedin Datasets working on top of its predecessor Folded Bipartite Network Embedding. The dataset that was used for this experiment was gathered through local machines scraping data on Linkedin.com. 



# Requirements 
The language that this project was done in was in Python 3.6 as most machine learning tools were commonly used in that language

It is recommended to create the environment using anaconda or by installing the packages listed in requirements.txt

`
pip install -r requirements.txt
`

# Datasets

As used in this project, we use different formats to process our Linkedin datasets to create our bipartite graph - there is 
 - User - Job
 - Company - Job
 - User - Company
 - User - Skill
 
 in this project we are mainly focusing on User to Skills and User to Jobs to best give the best recommendation for the user's future jobs
 
# Processes

There is 4 phases of processes that will be tested to determine how our model compares to other models. 

We have to test with the 
- original data preprocessing 
- original data preprocessing + LSTM (Long Short-Term Memory)
- our own data preprocessing that characterises a job/skill to a very generic type to make it easier to recommend and without LSTM (Long Short-Term Memory)
- our own data preprocessing + LSTM (Long Short-Term Memory)
 
# Evaluation

For evaluation, we test how long short-term memory has impacted our recommendations for our users, as well as how normalization the data will impact our recommendation with or without long short-term memory. We also tested against GraphRec and Folded Bipartite Network Embedding (FBNE) to determine how our implementation compares to the predecessors of the previous models. We focus on the mean reciprocal rank as our main point of metric to determine how it fares against the previous models. 

