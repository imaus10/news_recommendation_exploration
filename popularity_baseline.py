from pyspark.sql.functions import col

class PopularityBaseline:
    def fit(self, train):
        self.articles_by_popularity = train.groupBy('articleId') \
                                           .count() \
                                           .withColumnRenamed('count', 'prediction') \
                                           .checkpoint() # prevent recalculation every time
        return self
    def transform(self, to_predict):
        return to_predict.join(self.articles_by_popularity, on='articleId', how='left')
