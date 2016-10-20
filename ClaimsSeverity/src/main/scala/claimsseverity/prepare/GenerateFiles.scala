package claimsseverity.prepare

import claimsseverity.misc.SparkUtil._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

import scala.collection.immutable.IndexedSeq


class GenerateFiles {

  def generateTrainingFile(assembler: VectorAssembler, indexer: Option[Seq[StringIndexer]]) = {

    val trainFile = config.getString("claims.trainFile")

    val df = spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(trainFile)

    val indexedDf = indexer match {
      case Some(i) => i.foldLeft(df)((a,b) => b.fit(a).transform(a))
      case None => df
    }

    assembler.transform(indexedDf)

  }

  def generateTestingFile(assembler: VectorAssembler, indexer: Option[Seq[StringIndexer]]) = {
    val testFile = config.getString("claims.testFile")

    val df = spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(testFile)


    val indexedDf = indexer match {
      case Some(i) => i.foldLeft(df)((a,b) => b.fit(a).transform(a))
      case None => df
    }

     assembler.transform(indexedDf)

  }
}
