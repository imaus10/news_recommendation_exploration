import os
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, rand, row_number, sum

spark = SparkSession.builder.appName('NewsRecommendationExploration').getOrCreate()
checkpoint_dir = './data/spark-checkpoints'
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
spark.sparkContext.setCheckpointDir(checkpoint_dir)
# TODO: cleanup checkpoint dirs

def get_adressa_dataset():
    cleaned_filename = 'data/cleaned.json'
    if not os.path.exists(cleaned_filename):
        print('Loading full dataset...')
        # keep just article URLs. this drops a LOT of useless URLs.
        # from 10,043,269 rows down to 4,384,078
        article_views = spark.read.json('data/adressa_one_week') \
                                  .select('activeTime', 'url', 'userId') \
                                  .where(col('url').rlike('^http://(www\.)?adressa.no/.*\.(html|ece)(\?.*)?$'))

        # remove duplicates, summing the activeTime (cumulative time a user spent on an article)
        # down to 3,977,721
        article_views = article_views.groupBy(['userId', 'url']) \
                                     .agg(sum('activeTime').alias('activeTime')) \
                                     .checkpoint()

        # transform string IDs into numeric ID for ALS
        user_ids = article_views.select('userId').distinct() \
                                .withColumn('userIdNumeric', row_number().over(Window.orderBy('userId')))
        article_ids = article_views.select('url').distinct() \
                                   .withColumn('articleId', row_number().over(Window.orderBy('url')))
        article_views = article_views.join(user_ids, on='userId', how='left') \
                                     .join(article_ids, on='url', how='left') \
                                     .drop('userId', 'url') \
                                     .withColumnRenamed('userIdNumeric', 'userId') \
                                     .checkpoint()

        # users with only 1 view have no use for us -
        # if we train with them, we have no ground truth examples to evaluate our predictions.
        # we can collect them for a separate experiment on the cold start problem.
        cold_start_user_ids = article_views.groupBy('userId') \
                                           .count() \
                                           .where(col('count') < 2) \
                                           .select('userId') \
                                           .checkpoint()
        # left_anti means keep rows that do not appear in the right dataframe
        # down to 3,626,265
        article_views = article_views.join(cold_start_user_ids, on='userId', how='left_anti')

        # activeTime is blank for 2,242,418 rows. fill or drop?
        # for now: keep the rows, fill activeTime to a garbage placeholder value
        # TODO: what happens if we drop the rows instead?
        # or impute a value, like using the mean.
        # or set every value to 1 to create equal preference.
        article_views = article_views.fillna(0, subset=['activeTime']).checkpoint()
        article_views.write.json(cleaned_filename)
    else:
        print('Loading cleaned dataset from disk...')
        article_views = spark.read.json(cleaned_filename)

    return article_views

def get_data_split():
    train_filename = 'data/train.json'
    test_filename = 'data/test.json'
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        print('Loading train/test split from disk...')
        train = spark.read.json(train_filename)
        test = spark.read.json(test_filename)
    else:
        article_views = get_adressa_dataset()
        # for each user, remove one random article to use for testing
        # and put the rest into training
        # TODO: make k folds (k = avg articles/user?)
        user_views_numbered = article_views.withColumn('row_number', row_number().over(
            Window.partitionBy('userId').orderBy(rand())
        )).checkpoint()
        train = user_views_numbered.where(col('row_number') != 1) \
                                   .drop('row_number') \
                                   .checkpoint()
        # Remove targets that aren't in the training set
        # (don't penalize for something we couldn't know)
        article_ids = train.select('articleId').distinct()
        test = user_views_numbered.where(col('row_number') == 1) \
                                  .drop('row_number') \
                                  .join(article_ids, on='articleId', how='left_semi') \
                                  .checkpoint()
        # save the split to disk so we can iterate on multiple experiments
        train.write.json(train_filename)
        test.write.json(test_filename)
    return (train, test)
