## Overview

This is a basic exploration of news article recommendation in Spark.

It uses the Adressa dataset, which is a collection of event logs from a Norwegian news site. There are two sizes: one week and one month. This currently uses one week, which starts as 10,043,269 rows. After cleaning, there are 3,626,265 rows of a user viewing an article.

To compare methods, this computes click-thru rate (CTR). Each method predicts the top 20 articles for a given user. The test set holds one target article view out from each user's training data. If the target article for a user is in the method's 20 predicted articles, that counts as a click. To get CTR, we average the number of clicks over the number of users. This is pretty draconian, since we only get one shot to predict the right article.

Using CTR is expensive, because each user-article combination not in the training set has to be given a predicted preference weight. We could instead predict a single preference for the held-out articles and compare to a known preference value. But this doesn't really correspond to the reality of article recommendation, where we have thousands of possible articles to choose from and our job is to select the correct ones.

## Usage

```
# recommended: use a virtual env
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# download dataset
./download.sh

# get basic description of the full dataset in numbers
python explore.py

# run the methods
python main.py
```

## TODO:

- make computationally feasible
    - optimize queries
    - deploy on AWS cluster
- k fold validation (k = avg articles/user)
- show CTR as num of allowed predictions drops
- `activeTime` NA handling:
    - dropna
    - set all to common value (all article views equally weighted)
- scrape article content
- basic content-based recommendation:
    - remove stop words (Norwegian), collect dictionary
    - bag of words (binary word presence, word counts, and TF-IDF)
    - k-nearest neighbors to avg vector of all user's articles
- deep neural network content-based models:
    - word2vec or variants
    - multi-language models (e.g. LASER)
- hybrid collaborative/content methods
