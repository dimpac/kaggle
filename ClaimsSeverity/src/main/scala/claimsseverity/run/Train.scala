package claimsseverity.run

import claimsseverity.misc.SparkUtil._
import claimsseverity.train._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame

object Train {

  def main(args: Array[String]) {

    val trainDf = spark.read.format("parquet").load("tmp/trainDfStringIndexedFeatures.parquet")

    trainLinear(trainDf)
    trainDecisionTree(trainDf)
    trainRandomForest(trainDf)
    trainGBT(trainDf)

  }

  def trainDecisionTree(trainDf: DataFrame) = {
    val dtConfig = DecisionTreeConfig(maxBins = 512)
    val decisionTree = new DecisionTree("loss","features", dtConfig)

    val pipeline = new Pipeline()
      .setStages(Array(decisionTree.model))

    val model = pipeline.fit(trainDf)

    model.write.overwrite().save("tmp/decisionTree.model")
  }

  def trainLinear(trainDf: DataFrame) = {
    val config = LinearRegressionConfig()
    val lr = new Linear("loss","features", config)

    val pipeline = new Pipeline()
      .setStages(Array(lr.model))

    val model = pipeline.fit(trainDf)

    model.write.overwrite().save("tmp/linearRegression.model")
  }

  def trainRandomForest(trainDf: DataFrame) = {
    val rfConfig = RandomForestConfig(maxBins = 512)
    val randomForest = new RandomForest("loss","features", rfConfig)

    val pipeline = new Pipeline()
      .setStages(Array(randomForest.model))

    val model = pipeline.fit(trainDf)

    model.write.overwrite().save("tmp/randomForest.model")
  }

  def trainGBT(trainDf: DataFrame) = {
    val gbtConfig = GBTConfig(maxBins = 512)
    val gbt = new GradientBoostedTrees("loss","features", gbtConfig)

    val pipeline = new Pipeline()
      .setStages(Array(gbt.model))

    val model = pipeline.fit(trainDf)

    model.write.overwrite().save("tmp/gbt.model")
  }
}
