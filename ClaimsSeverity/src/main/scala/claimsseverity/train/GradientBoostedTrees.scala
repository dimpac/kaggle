package claimsseverity.train

import org.apache.spark.ml.regression.GBTRegressor

case class GBTConfig(maxBins: Int = 32, maxIter: Int = 2, maxDepth: Int = 4, impurity: String = "gini")

class GradientBoostedTrees(label: String, features: String, config: GBTConfig) {
  def model = {
    new GBTRegressor()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxIter(config.maxIter)
      .setMaxBins(config.maxBins)
      .setMaxDepth(config.maxDepth)
      .setImpurity(config.impurity)
  }
}
