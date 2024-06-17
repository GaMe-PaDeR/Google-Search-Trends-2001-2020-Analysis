from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofweek, hour, udf
from pyspark.sql.types import StringType
from pyspark.sql.types import TimestampType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
# from pyspark.ml.clustering import KMeans
from pyspark.ml.stat import Summarizer
import re
import findspark


# для создания пользовательских функций
from pyspark.sql.functions import udf 
# для использования оконных функций
from pyspark.sql.window import Window
# для работы с PySpark DataFrame
from pyspark.sql import DataFrame
# для задания типа возвращаемого udf функцией
from pyspark.sql.types import StringType
# для создания регулярных выражений
import re
# для работы с Pandas DataFrame
import pandas as pd
# для предобработки текста
from pyspark.ml.feature import HashingTF, IDF, Word2Vec,\
                               CountVectorizer, Tokenizer, StopWordsRemover
# для кластеризации
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import BisectingKMeans
# для создания пайплайна
from pyspark.ml import Pipeline
# для подсчета частоты слов в тексте
from nltk.probability import FreqDist


findspark.init()

# Инициализация SparkSession
spark = SparkSession.builder.appName("SearchQueryAnalysis").getOrCreate()

# Загрузка данных
yandex_df = spark.read.option("delimiter", "\t").csv("Google-Search-Trends-2001-2020-Analysis/football.dms", header=True).limit(10000)
yandex_df.show(truncate=False)

# Загрузка данных
categories_df = spark.read.csv("Google-Search-Trends-2001-2020-Analysis/trends.csv", header=True).limit(1000)
categories_df.show(truncate=False)

drop_list = ['location', 'year', 'rank']
categories_data = categories_df.select([column for column in categories_df.columns if column not in drop_list])
categories_data.show(5)

yandex_df = yandex_df.select(['normal_query'])
yandex_df.show(5)



def text_prep(text):
   # переводим текст в нижний регистр
    text = str(text).lower()
#    # убираем всё, что не русская буква, и убираем слово «баланс»
#     text = re.sub('[^а-яё]',' ',text)
#    # убираем всё, что начинается с «от»
#     text = re.sub('от.+','',text)
#    # убираем одиночные буквы
#     text = re.sub('\s[а-яё]\s{0,1}','',text)
   # если пробелов больше одного заменяем их на один
    text = re.sub('\s+',' ',text)
   # убираем лишние пробелы слева и справа
    text = text.strip()
    return text
# создание пользовательской функции
prep_text_udf = udf(text_prep, StringType())

yandex_df = yandex_df.withColumn("prep_query", prep_text_udf("normal_query")).filter('prep_query <> ""')

tokenizer = Tokenizer(inputCol = 'prep_query', outputCol = 'tokens')

# загрузим стоп-слова из pyspark
rus_stopwords = StopWordsRemover.loadDefaultStopWords('russian')
# загрузим локальные стоп-слова
with open('Google-Search-Trends-2001-2020-Analysis/stopwords.txt', 'r') as f:
    stopwords = [line.strip() for line in f]
# добавляем локальные стоп-слова к стоп-словам из pyspark
rus_stopwords.extend(stopwords)
# получим только уникальные значения из списка
rus_stopwords = list(set(rus_stopwords))
stopwordsRemover = StopWordsRemover(inputCol = 'tokens', 
                                    outputCol = 'clear_tokens', 
                                    stopWords = rus_stopwords)

hashingTF = HashingTF(inputCol = 'clear_tokens', outputCol = 'rawFeatures')
idf = IDF(inputCol = 'rawFeatures', outputCol = 'TfIdfFeatures', minDocFreq = 5)

word2Vec = Word2Vec(inputCol = 'clear_tokens', outputCol = 'Word2VecFeatures')

countVec = CountVectorizer(inputCol = 'clear_tokens', 
                           outputCol = 'CountVectFeatures')

pipeline = Pipeline(stages = [tokenizer, stopwordsRemover, 
  				     hashingTF, idf, word2Vec, countVec])
# применяем наш pipeline
pipeline_fit = pipeline.fit(yandex_df)
t = pipeline_fit.transform(yandex_df)


# import numpy as np
# # Calculate cost and plot
# cost = np.zeros(10)

# for k in range(2,10):
#     kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol('Word2VecFeatures')
#     model = kmeans.fit(t)
#     cost[k] = model.summary.trainingCost

# # Plot the cost
# df_cost = pd.DataFrame(cost[2:])
# df_cost.columns = ["cost"]
# new_col = [2,3,4,5,6,7,8, 9]
# df_cost.insert(0, 'cluster', new_col)

# import pylab as pl
# pl.plot(df_cost.cluster, df_cost.cost)
# pl.xlabel('Number of Clusters')
# pl.ylabel('Score')
# pl.title('Elbow Curve')
# pl.show()


cols = ['TfIdfFeatures', 'Word2VecFeatures', 'CountVectFeatures']

for col in cols:
    kmeans = KMeans().setK(36)\
               .setFeaturesCol(col)\
                     .setPredictionCol(f'kmeans_clusters_{col}')
    km_model = kmeans.fit(t)
    t = km_model.transform(t)

# for col in cols:
#     bkm = BisectingKMeans().setK(36)\
#                            .setFeaturesCol(col)\
#                            .setPredictionCol(f'BisectingKMeans_clusters_{col}')
#     bkm_model = bkm.fit(t)
#     t = bkm_model.transform(t)


col_clusters = ['kmeans_clusters_TfIdfFeatures',   
                'kmeans_clusters_Word2VecFeatures',
   		    'kmeans_clusters_CountVectFeatures']
# ,
 		    # 'BisectingKMeans_clusters_TfIdfFeatures',
  		    # 'BisectingKMeans_clusters_Word2VecFeatures',
  		    # 'BisectingKMeans_clusters_CountVectFeatures']

from pyspark.sql.functions import concat_ws
dataframes_list = []

for col in col_clusters:
    for i in range(36):
        ls = []
        tmp = t.select('clear_tokens',col).filter(f"{col} = {i}").collect()

        tmp = [tmp[j][0] for j in range(len(tmp))]

        for el in tmp:
            ls.extend(el)

        fdist = list(FreqDist(ls))[:5]

        d = {i:fdist}
        d = pd.DataFrame(list(d.items()), columns = [col, 'top_words'])
        d = spark.createDataFrame(d)

        d = d.withColumn('top_words_str', concat_ws(',', 'top_words')).drop('top_words')

        tmp_t = t.groupBy(col).count()\
                                .orderBy('count', ascending = False)\
                                .join(d, [col])

        # Добавляем DataFrame в список
        dataframes_list.append(tmp_t)

# Вывод всех DataFrames на экран
for df in dataframes_list:
    df.show(truncate=False)