from util import get_adressa_dataset
from pyspark.sql.functions import col, countDistinct, max, mean, min, stddev

if __name__ == '__main__':
    article_views = get_adressa_dataset()

    print('\nhow many user-article views? {}'.format(article_views.count()))

    print('\nhow many unique articles and users?')
    article_views.select(countDistinct('articleId'), countDistinct('userId')).show()

    print('\nwhat does the average user look like, in terms of number of article views?')
    article_views.groupBy('userId').count().describe(['count']).show()

    print('\nwhat\'s the spread of the view durations?')
    article_views.describe(['activeTime']).show()
