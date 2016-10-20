package claimsseverity.run

import claimsseverity.eda.BasicEda
import claimsseverity.misc.SparkUtil._


object Eda {

  def main(args: Array[String]) {
    val trainFile = config.getString("claims.trainFile")

    val df1 = spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(trainFile)

    val testFile = config.getString("claims.testFile")

    val df2 = spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(testFile)

    val basicEda = new BasicEda
    val trainEda = basicEda.nullColumns(df1)
    val testEda = basicEda.nullColumns(df2)

    println(s"Number of train dataset columns with nulls: ${trainEda.count(p => true)}")
    println(s"Number of test dataset columns with nulls: ${testEda.count(p => true)}")
  }

}
