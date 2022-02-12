val ScalaVersion = "2.11.12"

lazy val spark_mlib = "org.apache.spark" %% "spark-mllib" % "2.4.3"
lazy val spark_sql = "org.apache.spark" %% "spark-sql" % "2.4.3"

lazy val zoo =
  "com.intel.analytics.bigdl" % "bigdl-dllib-spark_2.4.6" % "0.14.0-SNAPSHOT"

lazy val gcn_with_zoo = project
  .in(file("."))
  .settings(
    name := "graph_conv_network",
    organization := "ems.gcn",
    scalaVersion := ScalaVersion,
    version := "0.0.1",
    libraryDependencies ++= Seq(
      spark_mlib,
      spark_sql,
      zoo
    )
  )

resolvers +=
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/groups/public/"
