package claimsseverity.train

import org.apache.spark.ml.regression.GBTRegressor

case class GBTConfig(maxBins: Int = 32, maxIter: Int = 16, maxDepth: Int = 10)

class GradientBoostedTrees(label: String, features: String, config: GBTConfig) {
  def model = {
    new GBTRegressor()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxIter(config.maxIter)
      .setMaxBins(config.maxBins)
      .setMaxDepth(config.maxDepth)
  }
}
