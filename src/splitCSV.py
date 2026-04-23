import pandas as pd
import os

if __name__=='__main__':
    base_dir = "/python/YOLO_traffic_flow/ICAE2024/maieeeaccess/Ariadna"
   
    df = pd.read_csv(os.path.join(base_dir,"output/perfMeasures/perfMeasures.csv"))

    dfK=df[df['Clustering method'] == 'KMEANS']
    dfMS=df[df['Clustering method'] == 'Mean Shift']
    dfDBSCAN=df[df['Clustering method'] == 'DBSCAN']

    dfK = dfK[['video', 'Number of clusters', 'MSE', 'DBI', 'Silhouette score', 'Calinsky-Harabasz']]
    dfMS = dfMS[['video', 'Number of clusters', 'MSE', 'DBI', 'Silhouette score', 'Calinsky-Harabasz']]
    dfDBSCAN = dfDBSCAN[['video', 'Number of clusters', 'MSE', 'DBI', 'Silhouette score', 'Calinsky-Harabasz']]


    


    print(dfK.to_latex(index = False))
    print(dfMS.to_latex(index = False))
    print(dfDBSCAN.to_latex(index = False))

    input()

    nameFileKMEANS = os.path.join(base_dir,"output/perfMeasures/perfMeasuresKMEANS.csv")
    nameFileMeanShift = os.path.join(base_dir,"output/perfMeasures/perfMeasuresMeanShift.csv")
    nameFileDBSCAN = os.path.join(base_dir,"output/perfMeasures/perfMeasuresDBSCAN.csv")

    dfK.to_csv(nameFileKMEANS)
    dfMS.to_csv(nameFileMeanShift)
    dfDBSCAN.to_csv(nameFileDBSCAN)