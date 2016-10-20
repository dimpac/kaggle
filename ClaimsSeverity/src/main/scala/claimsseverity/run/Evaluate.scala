package claimsseverity.run

import claimsseverity.misc.SparkUtil._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._


object Evaluate {

  def main(args: Array[String]) {
    val testDf = spark.read.format("parquet").load("tmp/testDfStringIndexedFeatures.parquet")
    val dtModel = "tmp/gbt.model"
    val columns = Seq(col("id"),col("prediction").as("loss"))
    val file = "tmp/output-GBT.csv"

    generateOutput(testDf, dtModel, columns, file)
  }

  def generateOutput(testDf: DataFrame, modelPath: String, columns: Seq[Column], file: String) = {
    val model = PipelineModel.load(modelPath)

    val predictions = model.transform(testDf)

    val outDf = predictions.select(columns:_*)

    outDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(file)
  }
}
