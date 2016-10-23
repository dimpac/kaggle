package claimsseverity.run

import claimsseverity.misc.SparkUtil._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._


object Evaluate {

  def main(args: Array[String]) {
    val testDf = spark.read.format("parquet").load(config.getString("claims.testFileCatIndexed"))
    //val dtModel = config.getString("claims.model.input.randomForest")
    val dtModel = config.getString("claims.model.input.gbt")
    val columns = Seq(col("id"),col("prediction").as("loss"))
    //val file = config.getString("claims.model.output.randomForest")
    val file = config.getString("claims.model.output.gbt")

    generateOutput(testDf, dtModel, columns, file)


  }

  def generateOutput(testDf: DataFrame, modelPath: String, columns: Seq[Column], file: String) = {

    //val model = PipelineModel.load(modelPath)

    val model = CrossValidatorModel.load(modelPath)
    val predictions = model.transform(testDf)

    val outDf = predictions.select(columns:_*)

    outDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(file)
  }
}
