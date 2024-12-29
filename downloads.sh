#!/bin/bash

# squad dataset
curl -L -o ./datasets/squad/stanford-question-answering-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/stanfordu/stanford-question-answering-dataset

# hotpotqa dataset
curl -L -o ./datasets/hotpotqa/hotpotqa-question-answering-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/jeromeblanchet/hotpotqa-question-answering-dataset