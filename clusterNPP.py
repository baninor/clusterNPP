# -*- coding: utf-8 -*-

#  4 Use a subset of only 3 classes from CIFAR10 (32x32 RGB images)

# Explore the database using unsupervised tools
# Try to cluster into 3 different clusters
# There is ground truth, but don’t use it for the fit, only for the evaluation and data exploration! (e.g. you can use Rand index)
# No need to split into Train & Test

#!pip install scikit-learn-extra
import sys
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
import statistics
from scipy.stats import hypergeom,ttest_1samp

import matplotlib
from matplotlib.pyplot import plot
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score

min_genes_per_cluster=5
max_genes_per_cluster=20

from  upgrade_clean_data import clean_df

param_grid = {
    'n_clusters': range (2,30)
}

###################save & restore pkl #####################
def save_ws_pkl(fname, listofOBJ):
    with open(fname, 'wb') as f:
        pickle.dump(listofOBJ, f)

def restore_ws_pkl(fname):
    with open(fname, 'rb') as f:
        listofOBJ = pickle.load(f)
    return (listofOBJ)
###################known pathways #####################
def load_known_pathways():
    fname='6_known_pathways_102023-2.csv'
    fpath = path.join('clades', 'known_pathways')
    fullpath = path.join(fpath, fname)
    knownPath=pd.read_csv(fullpath)#,sep='|',engine='python',encoding='latin-1') #'iso-8859-1' ,encoding='latin-1'
    s1='genes_no_medium_pg_no_low_scores'
    knownPath[s1] = knownPath[s1][:].map(lambda x: x.split(":"))
    knownPath = knownPath.rename(columns={'genes_no_medium_pg_no_low_scores': 'GenesChainu','gene_set_name':'name'})
    return knownPath

def load_WikiPathway_2021(dataset_genes,fname_org,fname4clade):
    WikiU=pd.DataFrame({'name': [] , 'GenesChainu': []})
    if  not os.path.isfile( fname4clade.replace('csv','pkl')):
        pg_table = pd.read_csv(fname_org)
        for i in range(pg_table.shape[0]):
            Pathway = list(pg_table.iloc[i])[0]
            num_genes = list(pg_table.iloc[i])[1]
            GenesChain = list(pg_table.iloc[i])[2]
            GenesList = GenesChain.split(";")
            GenesListu=list(set(GenesList) & set(dataset_genes))
            if len(GenesListu) < len(GenesList):
                print('was:', GenesList ,'upadte:',GenesListu)
                if len(GenesListu)==0:
                    continue
            WikiU=pd.concat([WikiU,pd.DataFrame({'name': [Pathway], 'GenesChainu': [GenesListu]})],ignore_index = False)
        WikiU.to_csv(fname4clade)
        save_ws_pkl( fname4clade.replace('csv','pkl'), [WikiU])

    else:
        WikiU=restore_ws_pkl(fname4clade.replace('csv','pkl'))[0]
    return WikiU
        #plot_gene_within_cluster(Pathway, GenesList)

def pval4known(genes_population_sz,cluster4pval,knownPath):
    pval2r=pd.DataFrame(columns=['knownP', 'Pval'])
    M=genes_population_sz
    genelist='genes_no_medium_pg_no_low_scores'
    for i in range(len(knownPath)):
        known_i=knownPath.iloc[i]
        knownlist=known_i['GenesChainu']
        n=len(knownlist)
        k=len(list(set(cluster4pval) & set(knownlist)))
        N=len(cluster4pval)
        prb = hypergeom.sf(k-1, M, n, N)
#        if prb < 1:
#            print ('Pval',prb,'knownPath',known_i['name'])
        pval2r=pd.concat( [pval2r,pd.DataFrame({'knownP':known_i['name'],'Pval':[prb]})],
                        ignore_index=True)
    idx=pval2r[pval2r['Pval']==pval2r['Pval'].min()].index[0]
    name=pval2r.iloc[idx]['knownP']
    p=pval2r.iloc[idx]['Pval']
    i=int(np.where(knownPath['name'] == name)[0])
    intersec_withKnown =list(set(knownPath.iloc[i]['GenesChainu'])&set(cluster4pval))
    return name,p,intersec_withKnown

def pval(population_sz,nlen,Nlen):
    # Suppose we have a collection of 20 animals, of which 7 are dogs.
    # Then if we want to know the probability of finding a given number of dogs
    # if we choose at random 12 of the 20 animals,
    # [M, n, N] = [20, 7, 12]
    # rv = hypergeom(M, n, N)
    # x = np.arange(0, n + 1)
    # pmf_dogs = rv.pmf(x)

    M=population_sz
    n=nlen
    k=n
    N=Nlen
    prb = hypergeom.sf(k-1, M, n, N)
    return prb

###########################silhouette ###############################

def modellist_above_max_Silhouette(silhouette4num_clusters):
    idx=np.where(np.array(silhouette4num_clusters['silhouette_avg']==silhouette4num_clusters['silhouette_avg'].max()))[0][0]
    print('thr', silhouette4num_clusters.iloc[idx]['distance_threshold'])
    return list(silhouette4num_clusters.iloc[idx:-1]['model']) #idx:-1

def model4max_Silhouette(silhouette4num_clusters):
    idx=np.where(np.array(silhouette4num_clusters['silhouette_avg']==silhouette4num_clusters['silhouette_avg'].max()))[0][0]
    print('thr', silhouette4num_clusters.iloc[idx]['distance_threshold'])
    return silhouette4num_clusters.iloc[idx]['model']

def isClusteralreadyexsist(clusterslist, cluster2check):
    if len(clusterslist)>0:
        d1=clusterslist[clusterslist['num_samples_in_cluster'] == cluster2check.iloc[0]['num_samples_in_cluster']]
        if len(d1)>0:
            for i in range(len(d1)):
                if set(d1.iloc[i]['listGenesInCluster'])==set(cluster2check.iloc[0]['listGenesInCluster']):
                    return True
    return False

def estimate_n_clusters_using_silhouette(X,fprfix,dstpath):
    #X- dataframe
    #fprfix - calde name
    # dst path for estimatetion result
    # Description: estimate number of distance_thr in the case of hirracial clustring distance='correlation'
    #return model with the value estimated ditance_thr and num_clusters=nan
    X1=X.copy()
    forcepkl=1
    silhouettefname=fprfix +'_'+'silhouette4_num_clusters'

    affinity = 'correlation'
    linkage = 'complete' #single,complete
    min_distance=0.05 #0.01
    max_distance=1.10 #1.99 1.10
    distance_step=0.05 #0.01
    silhouette_fullpath=os.path.join ('clades',dstpath,silhouettefname+str(min_distance)+'-'+str(max_distance)+'.pkl')

    param_grid = [ {'linkage': ['single','complete','average']}]
    #silhouette4num_clusters= pd.DataFrame({'distance_threshold': [], 'n_clusters_': [],'model': [],'silhouette_avg': []})
    dfcluster=pd.DataFrame()
    if not os.path.isfile(silhouette_fullpath) or forcepkl:
        clusters_list=[]
        for distance_threshold in np.arange(min_distance, max_distance, distance_step): # best was 0.35
            print('distance_threshold', distance_threshold)
            clustring_model = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None,
                                                      linkage=linkage, affinity=affinity)
            clustring_model.fit(X)
            X1['label']=clustring_model.labels_
            #R=plot_dendrogram(clustring_model)
            #if clustering_model.n_clusters_ < last_num_clusters:
            if len(np.unique( clustring_model.labels_)) > 1:
                silhouette_avg = silhouette_score(X, clustring_model.labels_, metric='correlation')
                sample_silhouette_values = silhouette_samples(X, clustring_model.labels_,metric='correlation')
                X1['SampSil']=sample_silhouette_values
                #for i in range(clustring_model.n_clusters_):
                for lbl in np.unique(clustring_model.labels_):
                   print('lbl',lbl)
                   SingleCluster = X1[X1['label'] == lbl].sort_values(by='SampSil', ascending=False)
                   num_samples_in_cluster= SingleCluster.shape[0]
                   # skip clusters with more than max_genes_per_cluster
                   if num_samples_in_cluster >max_genes_per_cluster or  \
                       num_samples_in_cluster < min_genes_per_cluster: # skip on clusters with less than 2 items...
                       continue
                   listGenesInCluster=list(SingleCluster.index)
                   silhoutte_sampls_values=list(SingleCluster['SampSil'])
                   X2=SingleCluster.copy()
                   X2.drop(['label', 'SampSil'],axis=1,inplace=True)
                   XcrossCorr=np.corrcoef(X2)
                   uper_triangle=XcrossCorr[np.triu(m=XcrossCorr, k=1) > 0]
                   AvgCrossCorr=uper_triangle.mean()
                   corr_med = statistics.median(uper_triangle)
                   corr_std = 0
                   if len(XcrossCorr) - 1 > 1:
                       corr_std = statistics.stdev(uper_triangle)
                   cluster2check=pd.DataFrame.from_dict(
                                        {'cluster_label': lbl,
                                         'distance_threshold': distance_threshold,
                                         'model': clustring_model,
                                         'num_samples_in_cluster':num_samples_in_cluster,
                                         'listGenesInCluster':[listGenesInCluster],
                                         'silhouette_avg': silhouette_avg,
                                         'CrossCorrMean': AvgCrossCorr,
                                         'CrossCorrStd': corr_std,
                                         'silhoutte_sampls_values':[silhoutte_sampls_values],
                                         'silhoutte_sampls_avg':X1[X1['label']==lbl]['SampSil'].mean()})
                   if isClusteralreadyexsist(dfcluster, cluster2check):
                       continue
                   dfcluster=pd.concat([dfcluster,cluster2check ], ignore_index=True)
            else:
                silhouette_avg=-1 # single label , silhouette not possible
        save_ws_pkl(silhouette_fullpath, [dfcluster])
    else:
        dfcluster = restore_ws_pkl(silhouette_fullpath)[0]
    m = model4max_Silhouette(dfcluster)
    models_list=modellist_above_max_Silhouette(dfcluster)
    return m,models_list,dfcluster
    #plot_distanceThr_vs_avg_var(cluster_by_distance)
    #clusters_with_filterd_size=filter_clusters_with_range_of_members(X, cluster_by_distance, selected_idx=0)


########################### Hierarchical_clustring ###############################

def rm_rows_with_zero_std(X):
    rows2rm=[]
    for i in range(X.shape[0]):
        if np.std(X.iloc[i])<10**-8:
            rows2rm.append(X.index[i])
    rmlbls=list(set(list(X.index)) - set(rows2rm))
    if len(rows2rm)>0:
        X=X.loc[rmlbls]
    return  X

def Hierarchical_clustring_by_correlation(X,clade,ResultsPath):
    X=rm_rows_with_zero_std(X)
    clustring_model,models_list,dfcluster=estimate_n_clusters_using_silhouette(X, fprfix=clade, dstpath=ResultsPath)
    cluster_info=pd.DataFrame()
    clusters_4range_dis=pd.DataFrame()
    cluster_infofname=clade +'_'+'cluster_info_correlation.pkl'

    cluster_info_fullpath=os.path.join ('clades',ResultsPath,cluster_infofname) #Hierarchical_by_corr_clusters

    affinity = 'correlation'
    linkage = 'complete' #single,complete

    # knownPath=load_known_pathways()
    fpathorg = os.path.join('clades', 'known_pathways')
    fpath = os.path.join('clades', 'known_pathways', 'not_lowscore')
    fname_org = os.path.join(fpathorg, 'WikiPathway_2021_Human_no_medium_pg_table.csv')
    fname4clade = os.path.join(fpath, clade + '_' + 'WikiPathway_2021_Human_no_medium_pg_table.csv')

    knownPath = load_WikiPathway_2021(dataset_genes=list(X.index), fname_org=fname_org, fname4clade=fname4clade)
    pd1 = pd.DataFrame({'WikiPathwayname': [], 'Pval': [], 'intersect': []})
    for i in dfcluster.index:
        name, p, intersec_withKnown = pval4known(genes_population_sz=X.shape[0],
                                                 cluster4pval=dfcluster.iloc[i]['listGenesInCluster'],
                                                 knownPath=knownPath)
        pd1 = pd.concat([pd1, pd.DataFrame({'WikiPathwayname': name, 'Pval': [p], 'intersect': [intersec_withKnown]})])
    pd1=pd1.reset_index().drop(labels='index',axis=1)
    dfcluster1=pd.concat([dfcluster,pd1],axis=1)
    dfcluster1.sort_values(by=['Pval','CrossCorrMean'], inplace=True, ascending=[True,False])
    dstpath=os.path.join('clades', ResultsPath)
    return dfcluster1 ,dstpath

################################### main #############################
matplotlib.use('TkAgg')
fname1='NPP_classic_no_medium_pg_no_low_score_genes.csv'
fname2='NPP_no_merdium_pg_no_low_score_genes.csv' # original
fname3='Homo_sapiens_NPP_UniProt_062018_Filtered_7057_genes_fixed.csv'
ResultsPath='Hierarchical_by_corr_clusters'


#fname='LPP_no_medium_pg_no_low_score_genes.csv' # LPP

#f={'infname':'NPP_no_merdium_pg_no_low_score_genes.csv','ResultsPath':'Hierarchical_by_corr_clusters'}
#f.append({'infname':'NPP_classic_no_medium_pg_no_low_score_genes.csv','ResultsPath':'Hierarchical_by_corr_clusters_0_not_modify'})
#f.append({'infname':'LPP_no_medium_pg_no_low_score_genes.csv','ResultsPath':'Hierarchical_by_corr_clusters_LPP'})
for fname in [fname3,fname2,fname1]:
    fname=os.path.join('NPP&LPP',fname)
    NPPfile=pd.read_csv(fname)
    NPPfile.rename(columns={"Unnamed: 0": "Gname"},inplace=True)
    list_of_clustring_alg=['Hierarchical_correlation'] # 'DBSCAN' 'Hierarchical_euclidean',
    ################## change nan values to newmin ##################
    NPPfile_new=NPPfile.copy()
    NPPfile=NPPfile.set_index('Gname')
    NPPfile=NPPfile.astype(float)
    #NPPfile=clean_df(X)
    ##########################################################
    if 'Hierarchical_correlation' in list_of_clustring_alg:
        clade_names=['Eukaryota']
        for clade in clade_names:
            #NPP_with_parasite = NPP_clade_only_with_parasite(NPPfile, [clade])
            if len(NPPfile)>0:
               #NPP_with_parasite=rm_low_score_genes(NPP_with_parasite,clade)
               dfcluster,dstpath=Hierarchical_clustring_by_correlation(NPPfile,clade,ResultsPath)
               distance_win = '_range_dist_' + str(round(dfcluster['distance_threshold'].min(), 2)) + '-' + str(
                   round(dfcluster['distance_threshold'].max(), 2))
               # add destination dir name for results
               orgfname=Path(os.path.normpath(fname).split(os.sep)[1]).stem
               dfcluster.to_csv(os.path.join(dstpath ,'Clusters_of_'+ orgfname+distance_win + '.csv'))

a=1

