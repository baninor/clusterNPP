Hierarchical clustering by correlation


Installation and running:
install packages: 
sklearn.cluster (AgglomerativeClustering)
sklearn.metrics (silhouette_samples, silhouette_score)
sklearn.feature_selection ( VarianceThreshold)

To use the code for creating clusters from NPP files:

1. Download "clusterNPP.py" and "upgrade_clean_data.py" in home directory
2.  Create in home directory folder "NPP&LPP" and locate the NPP file in this directory.
3. Create in home directory folder "clades\Hierarchical_by_corr_clusters" . 
4. "clusterNPP"   saves  clustering results in this folder using similar name (including      string with the cutting range of  the dendrogram) and with csv suffix .

5. In  "clusterNPP.py", verify that  "fname3"  is equal to the appropriate NPP filename
6. To create the Clusters csv file run "clusterNPP" 

 
Pseudo code
Hierarchical_clustering_by_correlation
for each threshold distance:
•	Perform Hierarchical_clustering
•	Compute the mean Silhouette Coefficient of all samples. 
        (average  Silhouette) for      specific threshold.
•	Calculate  Silhouette value per each  single cluster which includes  5 to 20 genes.
•	Per each single cluster: calculate "cross correlation mean"  
        and cross correlation std    for genes belonging to each cluster

 
The result is a sorted list of clusters with the following information for each cluster:
       dis_thr  - distance threshold)	
        id	 - cluster id
        num -number of genes in cluster	
        corr_med  -mean Silhouette Coefficient of all samples (in single distance threshold) 
        corr_std  ( cluster compactness – single clusrer)
        Gene_names (in cluster)	
        name (name of the best match entry from wiki- min pval on all wiki entries)
        Pval  (min pval)
        intersect (between Gene_names and genes of wiki entries)


Silhouette explanation below:
The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
This function returns the mean Silhouette Coefficient over all samples. To obtain the values for each sample, use silhouette_samples.
The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.


