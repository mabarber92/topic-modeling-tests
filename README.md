# topic-modeling-tests
Repo for Topic modelling test scripts, test corpora and outputs


## To do
- Wire up cluster_model in csvToBERTopic
- Experiment with mapping topic_model from one book to another - using [.update_topics](https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.update_topics) or [.transform](https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.transform)? Or [topics per class](https://maartengr.github.io/BERTopic/getting_started/topicsperclass/topicsperclass.html#saveload-bertopic-model) where class is a bookId?
- Research 'Online' approach to updating topic models for large corpora - [partial_fit()](https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.partial_fit) for identifying related topics? 
- Research [built in visualisation](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html)
- Attempt searching topics using [.find_topics()](https://maartengr.github.io/BERTopic/getting_started/search/search.html)
- Reorganise BERTopic - separating data, graphs and scripts
- Create an app.py that brings together the various functions into one piece that takes text files as an input
