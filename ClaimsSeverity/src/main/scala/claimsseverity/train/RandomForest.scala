package claimsseverity.train

import org.apache.spark.ml.regression.RandomForestRegressor

case class RandomForestConfig(maxBins: Int = 64, numTrees: Int = 4)

class RandomForest(label: String, features: String, config: RandomForestConfig) {
  def model = {
    new RandomForestRegressor()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxBins(config.maxBins)
      .setNumTrees(config.numTrees)
  }
}
