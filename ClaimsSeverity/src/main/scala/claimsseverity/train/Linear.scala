package claimsseverity.train

import org.apache.spark.ml.regression.LinearRegression


case class LinearRegressionConfig(maxIter: Int = 1, elasticNetParam: Double = 0.0)

class Linear(label: String, features: String, config: LinearRegressionConfig) {

  def model = {
     new LinearRegression()
      .setFeaturesCol(features)
      .setLabelCol(label)
      .setElasticNetParam(config.elasticNetParam)
      .setMaxIter(config.maxIter)

  }
}
