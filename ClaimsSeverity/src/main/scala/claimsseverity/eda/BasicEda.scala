package claimsseverity.eda

import org.apache.spark.sql.{DataFrame, Dataset}


class BasicEda {


  def nullColumns(df: DataFrame): Array[(String, Long)] = {

    val cols = df.columns

    val columnNullCount = cols.map(p => (p, df.where(df(p).isNull).count))

    columnNullCount.filter(p => p._2 != 0)
  }
}
