# Anime Recommendation Using MAL Database 2020 Dataset and Graph Machine Learning
Author: Zach Kangas

Date: 2/19/2023

Class: Graph Machine Learning

Professor: Dr. Urbain

# Dataset

https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020

# Abstract

For the longest time, matrix factorization was the go-to way to deal with huge recommender-type datasets. These matrix facorizations would essentially embed a large, extremely sparse matrix, perhaps containing hundreds of millions of entries, into a smaller dense matrix. The purpose of this experiment is to dive into the possiblity of using a node and edge type representation to turn a matrix problem into a graph problem. This is not a unique concept, but applied to the specific dataset, it seems to be one avenue that has not been explored yet.

Turning this dataset into a graph problem will allow more complex understandings and models with the benefit of improved prediction and training speeds, along with improved prediction performance. Part of this improvement in models is the use of embedded information within the various animes.

The matrix factorization model achieved a mean squared error score of 0.0173 after two epochs. The graph model achieved a mean squared error of 0.2752 on the validation set.

# Background

Recommender type systems are becoming extremely common in the modern era. From Twitter's endlessly scrolling front page to Google's AdSense, these are all systems that rely on prior information to recommend the user new and novel ideas, posts, and products. While the efficacy, transparency, and ethics of these systems is being highly debated as of now, it is stating a fact saying that these systems are becoming more and more common.

In this experiment, an anime recommendations database is analyzed. This is a relatively harmless example of a recommender system, given the only harm is long hours of binged anime, should it ever be implemented into a production system. This dataset contains about seventeen thousand Anime and the ratings coming from about 300,000 users. This amounts to about 57 million pairs of user to an anime rating. 

The nature of the dataset provides a significant problem for any computerized understanding of it. Given we can create an adjacency matrix using the dataset, we could hope to create a nearly five billion large entry table with dimensions 300,000 users by 17,000 animes. This adjacency matrix would take up over twenty gigabytes of memory at any given time, unless stored in a sparse format, and would be about 99% sparse, meaning 99% of the values would be null or similar representation.

Obvious to many is that this is not particularly feasible to do. While theoretically a AWS instance could be spun up with a GPU and 32+ GB of ram, there are better ways to solve this problem then by brute force.

The first of which is embedding the larger dataset space into a smaller one. Using matrix factorization, a large, extremely sparse represented matrix can be fit into a space of perhaps 32x32. This space contains most if not all the information from the original space, but then allows models to more efficiently perform predictions of a 32x32 space rather than a 17,000x300,000 space. A very similar if not identical concept is often used with natural language processing models, which want to embed the space of the vocabulary, often which is hundreds of thousands or millions words large, into a smaller, more manageable space.

This method of embedding this specific dataset has been done before multiple times. Kaggle has many linked notebooks which do this very thing.

The second way to mitigate this problem is by creating a graph representation of the dataset. There can be two types of nodes with one edge type. The first type of node would be animes, which could contain various information such as name, genre, or other information. The other node would be users, which would contain no information. Each user would hold edges to various anime that were weighted with a label.

The graph representation is what this experiment is attempting to create, then run a basic model off the information contained in the graph space. 

# Methods

As noted above, the dataset is an anime recommendations dataset scraped from MyAnimeList in 2020. This method has a few ramifications. Firstly, because it was scraped as well as being a public dataset, all users are anonymized. Users are instead represented with a serial integer instead of username or the MyAnimeList user id. Furthermore, MyAnimeList, even of a few years ago, had well over 5-10 million users, yet this dataset has 300,000 users represented. This is due to the method the original creator scraped the usernames. The usernames were obtained by using BeautifulSoup, a popular scraping addon for python, which would iterate over every single club MAL (MyAnimeList) had and pull all the usernames from any club with over 30 members. This means that users not very active on the platform, who would only update their scores as a method of keeping track of what they watched and when, are not properly represented in this dataset.

Early on, the dataset was attempted to be recreated. But was eventually abandoned since the API's that the original scraping script relied on are now either depreciated or not meant for any developer to pull from. While this could have been mitigated with a proper scraping script, scraping the specific information necessary to recreate the API would have required much effort and is instead left as a potential improvement to this experiment.

The dataset contains two files of note for this project. The first of which is `anime.csv` and the second is `rating_complete.csv`. The `ratings_complete.csv` is technically just a preprocessed format of `animelist.csv`. Where `animelist.csv` contains watching status (ie, completed, watching, on-hold, plan-to-watch), `ratings_complete.csv` filters for only status completed with a score that is not null. This makes it ideal for any basic model we attempt to create. However, it is important to note that the information provided in `animelist.csv` could have been used to create a more advanced model. If information about watching were included in a model, specifically which animes were dropped or on hold, it might have given a larger understanding into any given users habits and could have pulled information regarding what anime a user would like to watch better. However, this is left as a potential improvement to the experiment.

Due to the `ratings_complete.csv` being mostly preprocessed already, the only thing left to do to normalize the scores is change the ratings of 1-10 to another format. One possibility was to create nine separate categories, but instead a regression problem was created by doing `(user_score - 1) / 9`. This would give values between zero and one for scaling purposes

Each model described below was trained on Google Colab Pro using a High-RAM instance with a standard GPU accelerator. Each model was only trained on one-tenth of the total data. This is due to the fact that with even that amount of data, the models still took between 30 minutes to two hours to train. For sake a brevity for this experiment, much of the total data was left out.

The first model was a Matrix Factorization model. This model utilized Tensorflow and Keras. The purpose of this model was to give a baseline on what could be achieved with standard model architectures. The specific model had two embedding layers, one for users and one for anime. These layers would then be flattened and concatenated. From the flattened and concatenated model, two dense layers of 40 hidden nodes each analyzed the information each with a RelU activation, then one final output dense layer for the final prediction. There were nearly 2.6 million parameters for this model, most of which residing in the embedding layers.

The second model is a graph sage convolutional neural network. This model uses PyTorch Geometric. Processing the data for input took quite some time, but ended up with a graph representation as noted in the `Background` section. One extra thing to note is that the graph was turned into a undirected graph by creating identical but revered `anime rev_rates user` edges. This is done so that the model can traverse towards users from anime which will allow a better understanding of the graph as a whole. The purpose of the model was to perform edge-level predictions, and as such, the test and training data was split up by edges. The model at all points was information about some users but all anime, due to the edges contained in the training and testing data. This knowledge would most likely be the closest to what an actual production system might enjoy.

# Results

The final matrix factorization model took about 30 minutes to train over two epochs and resulted in a L2 loss of about 0.0173.

The final graph model took about 80 minutes to train over two epochs and resulted in a L2 loss of 0.2752.

# Conclusions

While a graph representation is wonderful, modern systems for embedding sparsity into dense matrices is actually extremely efficient. The keras model done for matrix factorization was extremely painless and only used some basic functional elements of keras to achieve a relatively complex model with non-standard forward propagation timings. The libraries for graph machine learning still seem to be in their relative infancy, with documentation insufficient compared to standard non-graph modeling like Tensorflow or PyTorch proper. However, even with that shortfall, the graph model performed well compared to the tried and true matrix factorization. The graph model can be improved by adding more tertiary features, including friends, watching status, and simply using the entire dataset. Further improvements can be made by using alternate networks and different hyper-parameters. More information could be embedded into the anime and user nodes to improve this model even further. The performance of this experiment was favorable, albeit lacking compared to the matrix factorization model. However, with more epochs and improved tuning, this new graph model could likely rival if not exceed the matrix factorization model.


# Sources and References:
https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70

https://medium.com/@yashsonar213/how-to-build-a-recommendation-systems-matrix-factorization-using-keras-778931cc666f

https://blog.dataiku.com/graph-neural-networks-link-prediction-part-two

https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html

https://medium.com/geekculture/how-to-load-a-dataset-from-the-google-drive-to-google-colab-67d0478bc634