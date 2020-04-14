from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from tqdm import tqdm
from popularity_baseline import PopularityBaseline
from util import get_data_split, spark

# returns a 1 or 0 depending on whether the given model
# contains the target article for the user in its top predictions
def get_user_hit(user_row, train, article_ids, model, num_predictions):
    user_id = user_row.userId
    target_article_id = user_row.articleId
    # the number of articles a given user has seen is pretty small (on the grand scale)
    already_seen = [row.articleId for row in train.where(col('userId') == user_id).collect()]
    # get predictions for every article not in the training set
    user_id_dataframe = spark.createDataFrame([user_row]).select('userId')
    to_predict = article_ids.where(~col('articleId').isin(already_seen)) \
                            .crossJoin(user_id_dataframe)
    predictions = model.transform(to_predict)
    top_predictions = predictions.orderBy(col('prediction').desc()).limit(num_predictions)
    return top_predictions.where(col('articleId') == target_article_id).limit(1).count()

if __name__ == '__main__':
    # TODO: vary this, graph dropoff
    # get rank of target article using row_number
    # and then .where(rank < num_predictions) to collect CTR
    num_predictions = 20

    (train, test) = get_data_split()
    methods = [
        PopularityBaseline(),
        # collaborative filtering - alternating least squares
        ALS(
            userCol='userId',
            itemCol='articleId',
            ratingCol='activeTime',
            implicitPrefs=True,
            coldStartStrategy='drop'
        )
    ]

    for method in methods:
        print('Training {} model'.format(method.__class__.__name__))
        model = method.fit(train)

        # cross join method (causes OOM error because it's billions of rows):
        # TODO: try broadcast()?
        # article_ids = train.select('articleId').distinct()
        # to_predict = test.select('userId') \
        #                  .crossJoin(article_ids) \
        #                  .exceptAll(train.select('userId', 'articleId'))
        # predictions = model.transform(to_predict)
        # top_predictions = predictions.withColumn('prediction_rank', row_number().over(
        #     Window.partitionBy('userId').orderBy(col('prediction').desc())
        # )).where(col('prediction_rank') <= num_predictions)
        # hits = test.select('userId', 'articleId') \
        #            .intersect(top_predictions.select('userId', 'articleId'))
        # click_thru_rate = hits.count() / test.count()

        # save this so it doesn't get computed for every user
        article_ids = train.select('articleId').distinct().checkpoint()
        # collecting all user ids wouldn't happen in production,
        # but we have to iterate over each user (due to OOM error above...)
        # Anyway, in production this would probably be a streaming operation over user events.
        print('predicting...')
        hits = [
            get_user_hit(user_row, train, article_ids, model, num_predictions)
            for user_row
            in tqdm(test.collect())
        ]
        click_thru_rate = sum(hits) / len(hits)
        print('click-thru-rate: {}'.format(click_thru_rate))
