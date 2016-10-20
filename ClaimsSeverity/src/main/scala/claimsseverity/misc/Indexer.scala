package claimsseverity.misc

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}


object Indexer {

  def stringIndexer(column: String, suffix: String) = {

    val outputColumn = column + suffix

    new StringIndexer()
      .setInputCol(column)
      .setOutputCol(outputColumn)
  }

  def assemble(input: Array[String], output: String) = {
    new VectorAssembler()
      .setInputCols(input)
      .setOutputCol(output)
  }

}
