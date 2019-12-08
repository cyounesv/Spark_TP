package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

import scala.collection.mutable.ArrayBuffer
import  org.apache.spark.ml.Pipeline


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    //val df = spark.read.parquet("src/main/ressources/prepared_trainingset")
    val df = spark.read.parquet("src/main/ressources/Fichiers-parquets")

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")
    val wordsData = tokenizer.transform(df)

    // remove stop words from file
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("removed")
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
    val cleanWordsData = remover.transform(wordsData)

    // TF IDF
    val countVect: CountVectorizerModel = new CountVectorizer()
      .setInputCol("removed")
      .setOutputCol("tf")
      .fit(cleanWordsData)
    val dfTest = countVect.transform(cleanWordsData)

    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")
    val idfModel = idf.fit(dfTest)
    val rescaledData = idfModel.transform(dfTest)

    // One hot encoding of country and currency
    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .fit(rescaledData)
    val rescaledDataCountry = indexerCountry.transform(rescaledData)

    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .fit(rescaledData)
    val rescaledDataCurrency = indexerCurrency.transform(rescaledDataCountry)

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))
      .fit(rescaledDataCurrency)
    val dfCategorical = oneHotEncoder.transform(rescaledDataCurrency)

    // Assembling all the features
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")
    val dfAssembled = assembler.transform(dfCategorical)

    // Lets use a logistic regression for our predictions
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    // Creation of the pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVect, idf, indexerCountry, indexerCurrency, oneHotEncoder, assembler, lr))

    // Splitting the data for train and evaluation
    val splits = df.randomSplit(Array(0.9, 0.1), seed=18)     // splitting training and test data
    val (trainingData, testData) = (splits(0), splits(1))

    pipeline.fit(trainingData)
    // Definition of our train validation param grid

    val elNetCvValues = getLogScale(1e-8, 10, 2)
    val minDFCvValues = Array(55.0, 75.0, 95.0)
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, elNetCvValues)
      .addGrid(countVect.minDF, minDFCvValues)
      .build()

    // Train validation
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setSeed(18)

    // Final predictions and scoring
    val cvModel = trainValidationSplit.fit(trainingData)
    val dfWithPredictions = cvModel.transform(testData)
    val f1SimpleScoreCV = cvModel.getEvaluator.evaluate(dfWithPredictions)
    println(f1SimpleScoreCV)
  }

  // useful functions to create log scale array of values
  private def getLogScale(from : Double, to : Double, logStep : Double):Array[Double] = {
    val initArray = new ArrayBuffer[Double]()
    initArray += from
    incrementLogScale(initArray, to, logStep).toArray
  }

  private def incrementLogScale(current : ArrayBuffer[Double], to : Double, logStep : Double):ArrayBuffer[Double] = {
    if (current.last < to){
      current += current.last*math.pow(10, logStep)
      incrementLogScale(current, to, logStep)
    }
    else{
      current
    }
  }

}