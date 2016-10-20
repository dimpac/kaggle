package claimsseverity.train

import org.apache.spark.ml.regression.DecisionTreeRegressor

case class DecisionTreeConfig(maxBins: Int)

class DecisionTree(label: String, features: String, config: DecisionTreeConfig) {

  def model: DecisionTreeRegressor = {
    new DecisionTreeRegressor()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxBins(config.maxBins)
  }
}
