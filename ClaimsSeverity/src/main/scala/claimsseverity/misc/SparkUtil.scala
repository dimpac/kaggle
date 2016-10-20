package claimsseverity.misc

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
  * Utility functions for seesion and configuration management
  */

object SparkUtil {
  lazy val config: Config = ConfigFactory.load()

  lazy val sparkConf = new SparkConf().setAll(configToMap(config.getConfig("core.config")))

  lazy val spark = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate()

  def configToMap(config: Config): Map[String, String] = {
    import scala.collection.JavaConversions._
    config.entrySet.map {
      case entry: Any => (entry.getKey, entry.getValue.unwrapped.toString)
    }.toMap
  }
}
