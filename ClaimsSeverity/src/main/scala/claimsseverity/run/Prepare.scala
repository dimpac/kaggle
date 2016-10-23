package claimsseverity.run


import claimsseverity.misc.Indexer._
import claimsseverity.prepare.GenerateFiles
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import claimsseverity.misc.SparkUtil._
// AKA ETL

object Prepare {

  def main(args: Array[String]) {

    val indexSuffix = "num"
    val cat = (1 to 116).map(p => s"cat$p")
    val suffixedCat = (1 to 116).map(p => s"cat${p}$indexSuffix")
    val cont = (1 to 14).map(p => s"cont$p")

    val assemblerInput = (suffixedCat ++ cont).toArray

    val idx = cat.map(p => stringIndexer(p, indexSuffix))

    val prepare = new GenerateFiles

    stringIndexedFeatures(prepare, assemble(assemblerInput, "features"), idx)

  }

  def stringIndexedFeatures(prepare: GenerateFiles, assembler: VectorAssembler, idx: Seq[StringIndexer]): Unit = {
    val trainDf = prepare.generateTrainingFile(assembler, Some(idx)).select("id","features","loss")
    trainDf.write.format("parquet").save(config.getString("claims.trainFileCatIndexed"))

    val testDf = prepare.generateTestingFile(assembler, Some(idx)).select("id","features")
    testDf.write.format("parquet").save(config.getString("claims.testFileCatIndexed"))
  }
}
