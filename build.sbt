val ScalaVersion = "2.12.15"

lazy val spark_mlib = "org.apache.spark" %% "spark-mllib" % "3.1.2"
lazy val spark_sql = "org.apache.spark" %% "spark-sql" % "3.1.2"

lazy val zoo = "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.13.0-spark_3.0.0" % "0.11.0"

lazy val  nlp_with_zoo= project
  .in(file("."))
  .settings(
    name := "graph_conv_network",
    organization := "ems.gcn",
    scalaVersion := ScalaVersion,
    version := "0.0.1",
    ThisBuild / useCoursier := false,
    libraryDependencies ++= Seq(
        spark_mlib, 
        spark_sql,
        zoo
        )
  )
