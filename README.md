# topic-modeling-tests
Repo for Topic modelling test scripts, test corpora and outputs


## To do
- Wire up cluster_model in csvToBERTopic
- Clean up BERTopic_model.py following tests for the corpus approach
- Document functions
- Created script for mapping a topic model to a series of texts - model needs to see all of text before transformation - now need to implement [partial_fit()](https://maartengr.github.io/BERTopic/getting_started/online/online.html#supervised-topic-modeling) to use an Online approach for iteratively training on a large corpus
- Research [built in visualisation](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html)
- Attempt searching topics using [.find_topics()](https://maartengr.github.io/BERTopic/getting_started/search/search.html)
- Reorganise BERTopic - separating data, graphs and scripts
- Create an app.py that brings together the various functions into one piece that takes text files as an input
