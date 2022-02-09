export SPARK_HOME=/home/ding/Downloads/spark-2.4.3-bin-hadoop2.7
MASTER=local[*] # the master url
${SPARK_HOME}/bin/spark-submit --master $MASTER --driver-memory 5g \
    --class ems.gcn.CoraExample \
      /home/ding/proj/SparkGCN/target/coraexample-0.1.0-SNAPSHOT-jar-with-dependencies.jar 1 1000 /home/ding/proj/SparkGCN/src/main/resources/data/cora.content /home/ding/proj/SparkGCN/src/main/resources/data/cora.cites
