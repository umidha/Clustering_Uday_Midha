import subprocess
import weka.core.jvm as jvm
from weka.core.converters import Loader 
from weka.core.converters import Saver
from weka.clusterers import Clusterer, ClusterEvaluation
import matplotlib.pyplot as plt
import weka.plot.dataset as pld
import weka.plot.clusterers as plc

## Downloading the data set
## If you don't have the data set in the current directory uncomment the part below

##path = "http://archive.ics.uci.edu/ml/machine-learning-databases/00290/eb.arff"
##subprocess.call(["curl","-O", path])

#################################################################

## Start the jvm
jvm.start()

## Load the data set and specify the class label coloumn
loader = Loader(classname = "weka.core.converters.ArffLoader")
saver = Saver(classname = "weka.core.converters.ArffSaver")
data = loader.load_file("eb.arff")

## Data Summary: The portion below does the first pass analysis of the data
print "Number of attributes:", data.num_attributes
print "Number of instances:", data.num_instances
print "Attributes:"
for i in range(0, data.num_attributes):
	print data.attribute(i)
	print data.attribute_stats(i)



####Uncomment the line below to plot the data points
####pld.scatter_plot(data, 0, 1)

#########Uncomment to save data with type as the last class
##data.delete_last_attribute()
##data.delete_last_attribute()
##saver.save_file(data, "data_with_class_type.arff")


### Deletes the not required attributes 
data.delete_attribute(2)
data.delete_attribute(2)
#####Uncomment to save the file with has serviceId as class, forkV and ForkW as attributes
###saver.save_file(data, "data_with_class_serviceID.arff")
data.delete_attribute(2)

#saver.save_file(data,"data.arff")
num_clusters = "6"   #Number of clusters for k mean

##Performing clustering
clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", num_clusters])
clusterer.build_clusterer(data)

for inst in data:
    cl = clusterer.cluster_instance(inst)  # 0-based cluster index
    dist = clusterer.distribution_for_instance(inst)   # cluster membership distribution
    #print("cluster=" + str(cl) + ", distribution=" + str(dist))

#########Getting the data about the clustered instances
evaluation = ClusterEvaluation()
evaluation.set_model(clusterer)
evaluation.test_model(data)
print evaluation.cluster_results
#print("# clusters: " + str(evaluation.num_clusters))
#print("log likelihood: " + str(evaluation.log_likelihood))
#print("cluster assignments:\n" + str(evaluation.cluster_assignments))
#plc.plot_cluster_assignments(evaluation, data,[],True)

####Using WEKA files to get the required results by calling them through this script

#########Calling the WEKA GUI to display the clusters
subprocess.call(["java" ,"-classpath", ".:weka.jar", "VisualizeClusterAssignments" ,"-t", "data.arff" ,"-W", "weka.clusterers.SimpleKMeans -N 6"]) ## Change the num_clusters here

#########Accuracy for clustering when target is serviceID
subprocess.call(["python", "clusterers.py", "-t", "data_with_class_serviceID.arff", "-c", "last", "weka.clusterers.SimpleKMeans", "-N", num_clusters])

#########Accuracy for clustering when target is type
subprocess.call(["python", "clusterers.py", "-t", "data_with_class_serviceID.arff", "-c", "last", "weka.clusterers.SimpleKMeans", "-N", num_clusters])
jvm.stop()
