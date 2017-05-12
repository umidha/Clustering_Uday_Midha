# Clustering_Uday_Midha

This repository contains the files and instructions for clustering the Tamilnadu Electricity Board Hourly Reading Data Set   
  
REQUIREMENTS:  
1. Python2.7
2. WEKA3.6.0.jar, rename as weka.jar place it within the folder where clustering.py is placed
3. Put all the files in the given repository to the location where clustering.py is placed

The file that automates the whole process is clustering.py    

To Visualize cluster assignments the script uses VisualizeCluseterAssignments.java obtained from https://weka.wikispaces.com/Visualizing+cluster+assignments   
  
To Compile VisualizeClusterAssignments.java use the following command   
javac -cp ./weka VisualizeClusterAssignments.java

To get the class accuracy results clusterers.py from WEKA is used  
