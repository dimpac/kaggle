package claimsseverity

import claimsseverity.misc.SparkUtil._
import claimsseverity.misc.Indexer._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, RandomForestRegressor}
import org.apache.spark.sql.functions._
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, DataUtils}
import scala.collection.immutable.IndexedSeq

object Main {

  def main(args: Array[String]) {

    val trainDf = spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("tmp/train.csv")

    val testDf = spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("tmp/test.csv")

    // 116 categorical variables
    // 14 numerical variables

    //Decision Trees
    //RMSE: 3019.71458
    val indexSuffix = "num"
    val cat = (1 to 116).map(p => s"cat$p")
    val suffixedCat = (1 to 116).map(p => s"cat${p}$indexSuffix")
    val cont = (1 to 14).map(p => s"cont$p")

    val x = cat.map(p => stringIndexer(p, indexSuffix))

    val trainDf2 = x.foldLeft(trainDf)((a,b) => b.fit(a).transform(a))

    val assemblerInput = (suffixedCat ++ cont).toArray

    val assembler = new VectorAssembler()
      .setInputCols(assemblerInput)
      .setOutputCol("features")

    val trainDf3 = assembler.transform(trainDf2)

    val train = trainDf3.withColumn("label", trainDf3("loss").cast("double")).select("label", "features")

    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap

    val xgboostModel = XGBoost.trainWithDataFrame(
      train, paramMap, 1, nWorkers = 1, useExternalMemory = true)

    /*val dt = new GBTRegressor()
      .setLabelCol("loss")
      .setFeaturesCol("features")
      .setMaxBins(512)
      .setMaxIter(20)*

    val model = dt.fit(trainDf3)
*/
    val testDf2 = x.foldLeft(testDf)((a,b) => b.fit(a).transform(a))

    val testDf3 = assembler.transform(testDf2)

    val test = testDf3.withColumn("label", lit(1.0)).select("id", "label", "features")

    val predictions = xgboostModel.setExternalMemory(true).transform(test)


   // val predictions = model.transform(testDf3)

    // Select example rows to display.
    val outDf = predictions.select(col("id"),col("prediction").as("loss"))

    outDf.write.format("com.databricks.spark.csv").save("tmp/output6.csv")

  }

}
