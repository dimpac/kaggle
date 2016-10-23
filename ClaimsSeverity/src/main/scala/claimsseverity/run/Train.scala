package claimsseverity.run

import claimsseverity.misc.SparkUtil._
import claimsseverity.train._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame

object Train {

  def main(args: Array[String]) {

    val trainDf = spark.read.format("parquet").load(config.getString("claims.trainFileCatIndexed"))

    //trainLinear(trainDf)
    //trainDecisionTree(trainDf)
    //trainRandomForest(trainDf)
    trainGBT(trainDf)

  }

  def trainDecisionTree(trainDf: DataFrame) = {
    val dtConfig = DecisionTreeConfig(maxBins = 512)
    val decisionTree = new DecisionTree("loss","features", dtConfig)

    val pipeline = new Pipeline()
      .setStages(Array(decisionTree.model))

    val model = pipeline.fit(trainDf)

    model.write.overwrite().save(config.getString("claims.model.input.decisionTree"))
  }

  def trainLinear(trainDf: DataFrame) = {
    val lrConfig = LinearRegressionConfig()
    val lr = new Linear("loss","features", lrConfig)

    val pipeline = new Pipeline()
      .setStages(Array(lr.model))

    val model = pipeline.fit(trainDf)

    model.write.overwrite().save(config.getString("claims.model.input.linearRegression"))
  }

  def trainRandomForest(trainDf: DataFrame) = {
    val rfConfig = RandomForestConfig(maxBins = 512, numTrees = 64)

    val nFolds = 10
    val randomForest = new RandomForest("loss","features", rfConfig)

    val pipeline = new Pipeline()
      .setStages(Array(randomForest.model))

    val paramGrid = new ParamGridBuilder().build() // No parameter search

    val evaluator = new RegressionEvaluator()
      .setLabelCol("loss")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(nFolds)

    val model = cv.fit(trainDf)

    model.write.overwrite().save(config.getString("claims.model.input.randomForest"))
  }

  def trainGBT(trainDf: DataFrame) = {
    val gbtConfig = GBTConfig(maxBins = 512)
    val nFolds = 10
    val gbt = new GradientBoostedTrees("loss","features", gbtConfig)

    val pipeline = new Pipeline()
      .setStages(Array(gbt.model))

    val paramGrid = new ParamGridBuilder().build() // No parameter search

    val evaluator = new RegressionEvaluator()
      .setLabelCol("loss")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(nFolds)

    val model = cv.fit(trainDf)

    model.write.overwrite().save(config.getString("claims.model.input.gbt"))
  }
}
