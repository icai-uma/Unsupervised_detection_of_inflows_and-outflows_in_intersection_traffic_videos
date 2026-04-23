import os
import glob
import csv
import cv2
import time
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.metrics import silhouette_score
from scriptHelpers import calculateClusteringPerformanceMeasures
from kneebow.rotor import Rotor
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import KElbowVisualizer

from pareto import *

seed = 123
random.seed(seed)
np.random.seed(seed)

mpl.rcParams['font.family'] = 'DejaVu Sans'

base_dir = "/python/YOLO_traffic_flow/ICAE2024/maieeeaccess/Ariadna"

postFix = '_age100'
postFix = ''

# Colores consistentes para todos los clusters
PALETA_COLORES = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#00FFFF', '#FE420F',
    '#FFFF14', '#C875C4',
    '#FF1493', '#00FF7F', '#FF4500', '#8A2BE2',
    '#A52A2A', '#5F9EA0', '#D2691E', '#FF6347', '#40E0D0', '#EE82EE',
    '#9ACD32', '#F4A460', '#6495ED', '#DC143C', '#00CED1', '#ADFF2F'
]

# ----------------------------- Detección y tracking -----------------------------
def save_initial_final_points_by_detection_tracking(output_folder:str,net=None, videolist = None, python_file:str = "yolov5demo.py",):
  CUDA=0
  PARAMS_CARLA="--img_size 1280 --track_points bbox --classes 2 3 7 6 --conf_thres 0.45 --iou_thresh 0.6 --numVehiculos 1000"
  PARAMS_REAL="--img_size 1280 --track_points bbox --classes 2 3 7 6 --conf_thres 0.45 --iou_thresh 0.6 --numVehiculos 1000"# --age 70"
  PARAMS_REAL2="--img_size 1280 --track_points bbox --classes 2 3 7 6 --conf_thres 0.45 --iou_thresh 0.6 --numVehiculos 1000"# --age 40"
  PARAMS_REAL_NORDIC="--img_size 1280 --track_points bbox --classes 2 3 7 6 --conf_thres 0.45 --iou_thresh 0.6 --numVehiculos 1000"# --age 70"
  if net is None:
    NET=('yolov5m6', 'yolov5x6', 'yolov5x')
  else:
    NET=(net,)

  videos=(
    ('videoDiagonal1', 'mp4', PARAMS_CARLA),
    ('videoDiagonal2', 'mp4', PARAMS_CARLA),
    ('videoHorizontal', 'mp4', PARAMS_CARLA),
    ('Highway', 'mp4', PARAMS_REAL),
    ('Seq1_SK_1', 'mp4', PARAMS_REAL),
    ('Seq1_SK_4', 'mp4', PARAMS_REAL),
    ('Seq2_SK_1', 'mp4', PARAMS_REAL),
    ('Seq2_SK_4', 'mp4', PARAMS_REAL),
    ('Seq3_SK_1', 'mp4', PARAMS_REAL),
    ('Seq3_SK_4', 'mp4', PARAMS_REAL),
    ('Seq3_SK_4', 'mp4', PARAMS_REAL2),
    ('Hadsundvej-1', 'mkv', PARAMS_REAL_NORDIC), 
    ('Hadsundvej-2', 'mkv', PARAMS_REAL_NORDIC), 
    ('Hasserisvej-1', 'mkv', PARAMS_REAL_NORDIC), 
    ('Hasserisvej-2', 'mkv', PARAMS_REAL_NORDIC), 
    ('Hasserisvej-3', 'mkv', PARAMS_REAL_NORDIC), 
    ('Hjorringvej-2', 'mkv', PARAMS_REAL_NORDIC), 
    ('Ostre-3', 'mkv', PARAMS_REAL_NORDIC)
  )

  # Filtrar videos si se pasa videolist
  if videolist is not None:
      videos = [v for v in videos if v[0] in videolist]
  
  #print(videos)

  for net in NET:
    for video, ext, params in videos:
        print(f"Applying detection and tracking methods to: '{video}'")
        os.system(f'export CUDA_VISIBLE_DEVICES={CUDA}; export VIDEO={video}; python {python_file} {base_dir}"/input/{video}.{ext}" {base_dir}"/{output_folder}/videos/{video}_{net}{postFix}.mp4" {base_dir}"/{output_folder}/points/{video}_{net}{postFix}.csv" --detector_path "{net}.pt" {params}')

# ----------------------------- Funciones auxiliares -----------------------------
def split_csv(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(base_dir, "output/perfMeasures/perfMeasures.csv"))
    
    # Filtrar por método de clustering
    dfGH = df[df['Clustering method'] == 'Geometric Heuristic']
    dfK = df[df['Clustering method'] == 'K-means (Elbow)']
    dfKS = df[df['Clustering method'] == 'K-means (Silhouette)']
    dfMS = df[df['Clustering method'] == 'Mean Shift']
    dfDBSCAN = df[df['Clustering method'] == 'DBSCAN']

    cols = df.columns.tolist()
    dfGH = dfGH[cols]
    dfK = dfK[cols]
    dfKS = dfKS[cols]
    dfMS = dfMS[cols]
    dfDBSCAN = dfDBSCAN[cols]

    # Guardar CSV
    files_csv = {
        "GH": dfGH,
        "kmeans": dfK,
        "kmeans_silhouette": dfKS,
        "MeanShift": dfMS,
        "DBSCAN": dfDBSCAN
    }

    for name, df_method in files_csv.items():
        csv_path = os.path.join(base_dir, f"output/perfMeasures/perfMeasures{name}.csv")
        txt_path = os.path.join(base_dir, f"output/perfMeasures/perfMeasures{name}_latex.txt")

        # Guardar CSV
        df_method.to_csv(csv_path, index=False)
        print(f"Saved: '{csv_path}'")

        # Guardar LaTeX en TXT
        with open(txt_path, "w") as f:
            f.write(df_method.to_latex(index=False,escape=True))
        print(f"Saved: '{txt_path}'")


def placeLegend(img, legendPath, legendPlace):
    # Leer la leyenda
    legend = cv2.imread(legendPath)
    if legend is None:
        print(f"Error: no se pudo leer la leyenda '{legendPath}'")
        return

    h_img, w_img = img.shape[:2]
    h_legend, w_legend = legend.shape[:2]

    # Posición inicial según legendPlace
    if legendPlace == 30:
        y_offset = h_img - h_legend
        x_offset = int(w_img/2 - w_legend/2)
    elif legendPlace == 23:
        y_offset = h_img - h_legend
        x_offset = w_img - w_legend
    elif legendPlace == 15:
        y_offset = int(h_img/2 - h_legend/2)
        x_offset = w_img - w_legend
    elif legendPlace == 60:
        y_offset = 0
        x_offset = int(w_img/2 - w_legend/2)
    elif legendPlace == 45:
        y_offset = int(h_img/2 - h_legend/2)
        x_offset = 0
    elif legendPlace == 37:
        y_offset = h_img - h_legend
        x_offset = 0
    elif legendPlace == 0:
        y_offset = int(h_img/2 - h_legend/2)
        x_offset = int(w_img/2 - w_legend/2)
    else:
        y_offset = int(h_img/2 - h_legend/2)
        x_offset = int(w_img/2 - w_legend/2)

    # Ajustar offsets negativos
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    # Recortar la leyenda si excede los límites de la imagen
    legend_h_end = min(y_offset + h_legend, h_img)
    legend_w_end = min(x_offset + w_legend, w_img)

    legend_crop_h = legend_h_end - y_offset
    legend_crop_w = legend_w_end - x_offset

    if legend_crop_h <= 0 or legend_crop_w <= 0:
        # La leyenda no cabe, no hacer nada
        return

    img[y_offset:legend_h_end, x_offset:legend_w_end] = legend[:legend_crop_h, :legend_crop_w]


def obtenerPrimerFrame(rutaVideo, rutaFrame):
  

  # Creamos un objeto de la clase VideoCapture que nos permitirá obtener los frames
  captura = cv2.VideoCapture(rutaVideo)
  
  # El método read() nos devuelve un boolean que nos indica si hay exito leyendo el frame, y el frame como tal
  success, frame = captura.read()
  
  if success:
    # Guardamos la imagen del frame en la ruta con su identificador
    os.makedirs(os.path.dirname(rutaFrame), exist_ok=True)
    cv2.imwrite(rutaFrame, frame)
    print(f"Saved first frame: '{rutaFrame}'")
    
  
  else:
    print("Error obteniendo el primer frame del vídeo")
  
  # Liberamos el objeto de VideoCapture
  captura.release()

def read_csv_points(csv_path):
    with open(csv_path) as f:
        X = np.genfromtxt(f, delimiter=",", skip_header=1)
    return X

def get_frame_size(primerFrame):
    img = cv2.imread(primerFrame)
    height, width = img.shape[:2]
    return width, height

def save_legend(ax, leyendaClusters, legendTextSize):
    leyenda = ax.legend(frameon=True)
    for item in leyenda.texts:
        item.set_fontsize(legendTextSize)
    fig = leyenda.figure
    fig.canvas.draw()
    bbox = leyenda.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    os.makedirs(os.path.dirname(leyendaClusters), exist_ok=True)
    fig.savefig(leyendaClusters, dpi="figure", bbox_inches=bbox)
    #print(f"Saved legend: '{leyendaClusters}'")

# ----------------------------- Funciones de clustering -----------------------------
def applyKmeans(csv_path, primerFrame, clustersPintados, leyendaClusters, legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net):
    start_time = time.perf_counter()

    X = read_csv_points(csv_path)
    
    rowForDataframe = [video, net]

    modelK = KMeans(n_clusters=4,n_init=10, random_state=0)
    labels_unique = np.arange(1)  # temporal
    # Visualizador KElbow (opcional)

    visualizer = KElbowVisualizer(modelK, timings=False, size=(1600, 1100))
    visualizer.fit(X)
    n_clusters_ = visualizer.elbow_value_
    

    modelK = KMeans(n_clusters=n_clusters_, n_init=10, random_state=0)
    modelK.fit(X)
    labels = modelK.labels_
    centroids = modelK.cluster_centers_
    labels_unique = np.unique(labels)

    runtime = (time.perf_counter() - start_time) * 1000

    plot_clusters(X, labels, centroids, primerFrame, clustersPintados, recortar)
    #plot_clusters(X, labels, centroids, primerFrame, clustersPintados, leyendaClusters, legendPlace, legendTextSize, recortar)
    
    rowForDataframe += ["K-means (Elbow)", len(labels_unique)]
    calculateClusteringPerformanceMeasures(X, labels, labels_unique, centroids,
                                           perfMeasures + "_kmeans.csv", rowForDataframe, dfResults, runtime)

def applyKmeans_silhouette(csv_path, primerFrame, clustersPintados, leyendaClusters,
                           legendPlace, legendTextSize, recortar,
                           perfMeasures, dfResults, video, net,
                           k_min=2, k_max=10):

    start_time = time.perf_counter()
    X = read_csv_points(csv_path)

    rowForDataframe = [video, net]

    best_k = None
    best_score = -1

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = model.fit_predict(X)

        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k

    # Fallback por seguridad
    if best_k is None:
        best_k = 4

    # Entrenamiento final con el mejor k
    modelK = KMeans(n_clusters=best_k, n_init=10, random_state=0)
    modelK.fit(X)

    labels = modelK.labels_
    centroids = modelK.cluster_centers_
    labels_unique = np.unique(labels)

    runtime = (time.perf_counter() - start_time) * 1000

    plot_clusters(X, labels, centroids, primerFrame, clustersPintados, recortar)
    #plot_clusters(X, labels, centroids, primerFrame, clustersPintados, leyendaClusters, legendPlace, legendTextSize, recortar)

    rowForDataframe += ["K-means (Silhouette)", len(labels_unique)]

    calculateClusteringPerformanceMeasures(
        X, labels, labels_unique, centroids,
        perfMeasures + "_kmeans_silhouette.csv",
        rowForDataframe, dfResults, runtime
    )

def applyMeanShift(csv_path, primerFrame, clustersPintados, leyendaClusters, legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net):
    start_time = time.perf_counter()
    
    X = read_csv_points(csv_path)
    rowForDataframe = [video, net]

    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    centroids = ms.cluster_centers_
    labels_unique = np.unique(labels)
    
    runtime = (time.perf_counter() - start_time)* 1000

    plot_clusters(X, labels, centroids, primerFrame, clustersPintados, recortar)
    #plot_clusters(X, labels, centroids, primerFrame, clustersPintados, leyendaClusters, legendPlace, legendTextSize, recortar)

    rowForDataframe += ["Mean Shift", len(labels_unique)]
    calculateClusteringPerformanceMeasures(X, labels, labels_unique, centroids,
                                           perfMeasures + "_meanShift.csv", rowForDataframe, dfResults, runtime)

def applyDBSCAN(csv_path, primerFrame, clustersPintados, leyendaClusters,legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net):
    start_time = time.perf_counter()
    X = read_csv_points(csv_path)
    rowForDataframe = [video, net]

    eps = estimate_dbscan_eps(X)
    clusters = DBSCAN(eps=eps, min_samples=4).fit(X)
    labels = clusters.labels_
    labels_unique = np.unique(labels)
    
    centroids = estimate_centroids(X, labels)

    runtime = (time.perf_counter() - start_time)* 1000
    
    plot_clusters(X, labels, centroids, primerFrame, clustersPintados, recortar)
    #plot_clusters(X, labels, centroids, primerFrame, clustersPintados, leyendaClusters, legendPlace, legendTextSize, recortar)

    rowForDataframe += ["DBSCAN", len(labels_unique)]
    calculateClusteringPerformanceMeasures(X, labels, labels_unique, centroids,
                                           perfMeasures + "_DBSCAN.csv", rowForDataframe, dfResults, runtime)

def applyGeometricHeuristic(csv_path, primerFrame, clustersPintados, leyendaClusters,legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net):
    start_time = time.perf_counter()
    X = read_csv_points(csv_path)
    rowForDataframe = [video, net]
    
    width, height = get_frame_size(primerFrame)
    x0, y0 = width/2, height/2

    def get_cluster(xi, yi):
        if xi < x0 and yi < y0: return 0
        elif xi >= x0 and yi < y0: return 1
        elif xi < x0 and yi >= y0: return 2
        else: return 3

    labels = np.array([get_cluster(xi, yi) for xi, yi in X])
    labels_unique = np.unique(labels)
    centroids = np.array([X[labels==lab].mean(axis=0) for lab in labels_unique]) # se queda con los puntos de cada cuadrante y calcula el punto medio (centroide)

    runtime = (time.perf_counter() - start_time)* 1000
    
    plot_clusters(X, labels, centroids, primerFrame, clustersPintados, recortar)
    #plot_clusters(X, labels, centroids, primerFrame, clustersPintados, leyendaClusters, legendPlace, legendTextSize, recortar)

    rowForDataframe += ["Geometric Heuristic", len(labels_unique)]
    calculateClusteringPerformanceMeasures(X, labels, labels_unique, centroids,
                                           perfMeasures + "_gh.csv", rowForDataframe, dfResults, runtime)

# ----------------------------- Funciones auxiliares de DBSCAN -----------------------------
def calculate_kn_distance(X, neigh=20):
    neighbors = NearestNeighbors(n_neighbors=neigh)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    return distances[:,1:].reshape(-1)

def estimate_dbscan_eps(X, neigh=4):
    eps_dist = np.sort(calculate_kn_distance(X, neigh))
    rotor = Rotor()
    curve_xy = np.concatenate([np.arange(eps_dist.shape[0]).reshape(-1,1), eps_dist.reshape(-1,1)],1)
    rotor.fit_rotate(curve_xy)
    e_idx = rotor.get_elbow_index()
    return curve_xy[e_idx,1]

def estimate_centroids(X, labels):
    labels_unique = np.unique(labels)
    centroids = np.array([
        X[labels == lab].mean(axis=0)
        for lab in labels_unique if lab != -1
    ])
    return centroids


# ----------------------------- Función de plot común -----------------------------

PALETA_COLORES = [
    (180,119,31), (14,127,255), (44,160,44), (40,39,214),
    (189,103,148), (75,86,140), (194,119,227), (127,127,127),
    (34,189,188), (207,190,23),
    (255,127,14), (148,103,189), (23,190,207), (140,75,86),
    (255,158,74), (199,0,57), (128,0,128), (0,128,128)
]

def font_scale_latex(img, ancho_ref=1200, scale_ref=0.7):
    _, w = img.shape[:2]
    return scale_ref * (w / ancho_ref)

def drawLegendCV(
    labels,
    colors,
    marker_radius,
    font_scale,
    line_height,
    padding=10,
    bg_color=(255, 255, 255)
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    text_x = padding + marker_radius * 2 + 15

    widths = [
        cv2.getTextSize(lbl, font, font_scale, thickness)[0][0]
        for lbl in labels
    ]

    w = text_x + max(widths) + padding
    h = padding * 2 + line_height * len(labels)

    legend = np.full((h, w, 3), bg_color, dtype=np.uint8)

    for i, lbl in enumerate(labels):
        y = padding + i * line_height + line_height // 2
        color = tuple(int(c) for c in colors[i % len(colors)])

        cv2.circle(
            legend,
            (padding + marker_radius, y),
            marker_radius,
            color,
            -1
        )

        cv2.putText(
            legend,
            lbl,
            (text_x, y + int(0.35 * line_height)),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    return legend

def placeLegendOutside(img, legend, margin=20, bg_color=(255, 255, 255)):
    h_img, w_img = img.shape[:2]
    h_leg, w_leg = legend.shape[:2]

    new_h = max(h_img, h_leg)
    new_w = w_img + margin + w_leg

    canvas = np.full((new_h, new_w, 3), bg_color, dtype=np.uint8)
    canvas[0:h_img, 0:w_img] = img

    y_offset = (new_h - h_leg) // 2
    x_offset = w_img + margin

    canvas[y_offset:y_offset + h_leg, x_offset:x_offset + w_leg] = legend

    return canvas

def pintarPuntosTrayectoria(
    rutaImagen,
    arrayDatos,
    centroides,
    labelClusters,
    rutaResultado,
    nombresClusters=None,
    recortar=None
):
    img = cv2.imread(rutaImagen)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen '{rutaImagen}'")

    # Escalado global coherente para LaTeX
    font_scale = font_scale_latex(img)
    k = font_scale / 0.7

    s = int(16 * k)

    # Dibujar puntos
    for f in range(arrayDatos.shape[0]):
        x = int(float(arrayDatos[f][0]))
        y = int(float(arrayDatos[f][1]))
        color = tuple(int(c) for c in PALETA_COLORES[labelClusters[f] % len(PALETA_COLORES)])
        cv2.circle(img, (x, y), int(round(s * 0.75)), color, -1)

    # Dibujar centroides
    for n in range(centroides.shape[0]):
        x = int(float(centroides[n][0]))
        y = int(float(centroides[n][1]))
        cv2.circle(img, (x, y), s, (0, 0, 0), -1)

    # Recorte opcional
    if recortar is not None:
        img = img[recortar[0]:recortar[2], recortar[1]:recortar[3], :]

    # -------- Guardar imagen SIN leyenda --------
    img_sin_leyenda = img.copy()
    rutaResultadoSinLeyenda = os.path.join(os.path.dirname(rutaResultado),"nolegend")
    os.makedirs(rutaResultadoSinLeyenda, exist_ok=True)
    rutaResultadoSinLeyenda = os.path.join(rutaResultadoSinLeyenda,os.path.basename(rutaResultado))
    cv2.imwrite(
        rutaResultadoSinLeyenda,
        img_sin_leyenda,
        [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )

    # -------- Crear imagen CON leyenda --------
    if nombresClusters is None:
        nombresClusters = [f"Cluster {i}" for i in range(centroides.shape[0])]

    legend = drawLegendCV(
        nombresClusters,
        PALETA_COLORES,
        marker_radius=int(round(s * 0.75)),
        font_scale=font_scale,
        line_height=int(40 * k)
    )

    img_con_leyenda = placeLegendOutside(img, legend)

    cv2.imwrite(
        rutaResultado,
        img_con_leyenda,
        [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )

    # -------- Guardar centroides en CSV --------
    rutaDir = os.path.dirname(os.path.dirname(os.path.dirname(rutaResultado)))
    rutaCentroides = os.path.join(rutaDir, "centroids")
    os.makedirs(rutaCentroides, exist_ok=True)

    nombreCSV = os.path.basename(rutaResultado.replace("_ClustersPintados","").replace(".png","")) + "_centroids.csv"
    rutaCSV = os.path.join(rutaCentroides, nombreCSV)

    with open(rutaCSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "x", "y"])
        for i in range(centroides.shape[0]):
            x = float(centroides[i][0])
            y = float(centroides[i][1])
            writer.writerow([i, x, y])
    #print(f"Saved: {rutaCSV}")

def plot_clusters(
    X,
    labels,
    centroids,
    primerFrame,
    clustersPintados,
    clustersPintadosSinLeyenda,
    recortar=None
):
    cluster_ids = [lab for lab in np.unique(labels) if lab != -1]
    nombresClusters = [f"Cluster {k}" for k in cluster_ids]

    pintarPuntosTrayectoria(
        rutaImagen=primerFrame,
        arrayDatos=X,
        centroides=centroids,
        labelClusters=labels,
        rutaResultado=clustersPintados,
        nombresClusters=nombresClusters,
        recortar=recortar
    )

# ----------------------------- Función de evaluación por cada método -----------------------------
def scriptEv(net=None, videolist=None, output_folder:str = "output"):
    import pandas as pd
    
    d = {
        "video": [], "network": [], "Clustering method": [], 
        "Number of clusters": [], "MSE": [], "DBI": [], 
        "Silhouette score": [], "Calinski-Harabasz": [],
        "Runtime (ms)":[]
    }
    dfResults = pd.DataFrame(d)

    rec1 = [0, 239, -1, 1680]  # [corner1 X, corner1 Y, corner2 X, corner2 Y]

    # Lista de videos
    videos = (
    ('videoDiagonal1', 'mp4', 30, 40, None),
    ('videoDiagonal2', 'mp4', 37, 40, None),
    ('videoHorizontal', 'mp4', 30, 40, None),
    ('Highway', 'mp4', 15, 40, rec1),
    ('Seq1_SK_1', 'mp4', 30, 40, rec1),
    ('Seq1_SK_4', 'mp4', 37, 40, rec1),
    ('Seq2_SK_1', 'mp4', 37, 40, rec1),
    ('Seq2_SK_4', 'mp4', 37, 40, rec1),
    ('Seq3_SK_1', 'mp4', 0, 40, rec1),
    ('Seq3_SK_4', 'mp4', 45, 40, rec1),
    ('Hadsundvej-1', 'mkv', 23, 20, None), 
    ('Hadsundvej-2', 'mkv', 23, 20, None), 
    ('Hasserisvej-1', 'mkv', 23, 20, None), 
    ('Hasserisvej-2', 'mkv', 23, 20, None), 
    ('Hasserisvej-3', 'mkv', 23, 20, None), 
    ('Hjorringvej-2', 'mkv', 23, 20, None), 
    ('Ostre-3', 'mkv', 23, 20, None)
    )

    # Filtrar videos si se pasa videolist
    if videolist is not None:
        videos = [v for v in videos if v[0] in videolist]
        #print(videos)
    # Lista de redes
    if net is None:
        nets = ('yolov5m6', 'yolov5x6', 'yolov5x')
    else:
        nets = (net,)

    for net in nets:
        print(f"Processing network: '{net}'")
        for video, ext, legendPlace, legendTextSize, recortar in videos:
            print(f"Processing video: '{video}'")
            
            name = f"{video}_{net}" 
            inpv = f"{base_dir}/input/{video}.{ext}"
            csv = f"{base_dir}/{output_folder}/points/{name}.csv"
            pframe = f"{base_dir}/{output_folder}/first_frame/{name}_primerFrameVideo.png"
            clustersp = f"{base_dir}/{output_folder}/figures/clusters/{name}_ClustersPintados_"
            lclusters = f"{base_dir}/{output_folder}/figures/legends/{name}_LeyendaClusters_"
            perfMeasures = f"{base_dir}/{output_folder}/perfMeasures/{name}_perfMeasures"

            os.makedirs(os.path.dirname(pframe), exist_ok=True)
            os.makedirs(os.path.dirname(clustersp), exist_ok=True)
            os.makedirs(os.path.dirname(lclusters), exist_ok=True)
            os.makedirs(os.path.dirname(perfMeasures), exist_ok=True)

            # Obtener primer frame
            obtenerPrimerFrame(inpv, pframe)
            
            # Aplicar clustering
            applyGeometricHeuristic(csv, pframe, clustersp+"GH.png", lclusters+"GH.png", legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net)
            applyKmeans(csv, pframe, clustersp+"kmeans.png", lclusters+"kmeans.png", legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net)
            applyKmeans_silhouette(csv, pframe, clustersp+"kmeans_silhoutte.png", lclusters+"kmeans_silhoutte.png", legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net)
            applyMeanShift(csv, pframe, clustersp+"meanshift.png", lclusters+"meanshift.png", legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net)
            applyDBSCAN(csv, pframe, clustersp+"DBSCAN.png", lclusters+"DBSCAN.png", legendPlace, legendTextSize, recortar, perfMeasures, dfResults, video, net)

            # Guardar resultados después de cada video
            perfMeasuresCSV = f"{base_dir}/{output_folder}/perfMeasures/perfMeasures.csv"
            os.makedirs(os.path.dirname(perfMeasuresCSV), exist_ok=True)
            dfResults.to_csv(perfMeasuresCSV, index=False)
    
    print("-"*150)
    print(dfResults)
    print("-"*150)
    print(f"Saved: '{perfMeasuresCSV}'")

    txt_perfMeasures = perfMeasuresCSV.replace(".csv","_latex.txt")
    with open(txt_perfMeasures, "w") as f:
      f.write(dfResults.to_latex(index=False,escape=True))
    print(f"Saved: '{txt_perfMeasures}'")

    split_csv(dfResults)
    
def split_by_metric(dfResults, output_dir):
    import os
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    metricas = [
        'Number of clusters', 'MSE', 'DBI',
        'Silhouette score', 'Calinski-Harabasz', 'Runtime (ms)'
    ]
    
    orden_metodos = [
        'Geometric Heuristic',
        'K-means (Elbow)',
        'K-means (Silhouette)',
        'Mean Shift',
        'DBSCAN'
    ]
    
    maximize_metrics = ['Silhouette score', 'Calinski-Harabasz']
    minimize_metrics = ['MSE', 'DBI', 'Runtime (ms)']
    
    orden_videos = dfResults['video'].drop_duplicates().tolist()
    tablas_generadas = {}
    
    for metrica in metricas:
        # Crear tabla vacía
        df_tabla = pd.DataFrame(
            index=orden_videos,
            columns=orden_metodos,
            dtype=float
        )
        
        for _, fila in dfResults.iterrows():
            video = fila['video']
            metodo = fila['Clustering method']
            
            if video in df_tabla.index and metodo in df_tabla.columns:
                df_tabla.at[video, metodo] = fila[metrica]
        
        df_tabla.loc['Mean'] = df_tabla.mean()
        df_tabla.loc['Std'] = df_tabla.std()
        
        # Guardar CSV
        csv_path = os.path.join(
            output_dir,
            f"tabla_{metrica.replace(' ', '_')}.csv"
        )
        df_tabla.to_csv(csv_path)
        print(f"Saved CSV: '{csv_path}'")
        
        
        df_tabla = df_tabla.round(3)

        # Preparar LaTeX
        df_latex = df_tabla.copy().astype(object)
        
        for idx, row in df_tabla.iterrows():
            if metrica in maximize_metrics:
                mejor = row.max()
            elif metrica in minimize_metrics:
                mejor = row.min()
            else:
                mejor = None
            
            for col in df_tabla.columns:
                if pd.isna(row[col]):
                    df_latex.at[idx, col] = ""
                    continue
                
                if mejor is not None and row[col] == mejor:
                    df_latex.at[idx, col] = f"$\\boldsymbol{{{row[col]:.3f}}}$"
                else:
                    df_latex.at[idx, col] = f"{row[col]:.3f}"
        
        latex_tabla = df_latex.to_latex(
            index=True,
            escape=False,
            column_format='c' * (len(df_latex.columns) + 1)
        )
        
        latex_tabla = latex_tabla.replace("_","\_")

        header = (
            "Video & Quadrant-based & K-means (Elbow) & "
            "K-means (Silhouette) & Mean Shift & DBSCAN \\\\\n"
        )
        
        latex_tabla = latex_tabla.splitlines()
        latex_tabla[2] = header
        latex_tabla = "\n".join(latex_tabla)
        
        # Envolver en table*
        latex_tabla = (
            "\\begin{table*}[ht]\n"
            "\\centering\n"
            f"\\caption{{{metrica}}}\n"
            + latex_tabla +
            "\n\\end{table*}\n"
        )
        
        txt_path = os.path.join(
            output_dir,
            f"tabla_{metrica.replace(' ', '_')}.txt"
        )
        with open(txt_path, "w") as f:
            f.write(latex_tabla)
        
        print(f"Saved LaTeX (txt): '{txt_path}'")
        tablas_generadas[metrica] = df_tabla

def paint_centroids(net=None, videolist=None, output_folder:str = "output_rereview"):
    methods = ["DBSCAN","GH", "kmeans", "kmeans_silhoutte","meanshift"]
    rec1 = [0, 239, -1, 1680]  # [corner1 X, corner1 Y, corner2 X, corner2 Y]

    # Lista de videos
    videos = (
    ('videoDiagonal1', 'mp4', 30, 40, None),
    ('videoDiagonal2', 'mp4', 37, 40, None),
    ('videoHorizontal', 'mp4', 30, 40, None),
    ('Highway', 'mp4', 15, 40, rec1),
    ('Seq1_SK_1', 'mp4', 30, 40, rec1),
    ('Seq1_SK_4', 'mp4', 37, 40, rec1),
    ('Seq2_SK_1', 'mp4', 37, 40, rec1),
    ('Seq2_SK_4', 'mp4', 37, 40, rec1),
    ('Seq3_SK_1', 'mp4', 0, 40, rec1),
    ('Seq3_SK_4', 'mp4', 45, 40, rec1),
    ('Hadsundvej-1', 'mkv', 23, 20, None), 
    ('Hadsundvej-2', 'mkv', 23, 20, None), 
    ('Hasserisvej-1', 'mkv', 23, 20, None), 
    ('Hasserisvej-2', 'mkv', 23, 20, None), 
    ('Hasserisvej-3', 'mkv', 23, 20, None), 
    ('Hjorringvej-2', 'mkv', 23, 20, None), 
    ('Ostre-3', 'mkv', 23, 20, None)
    )

    # Filtrar videos si se pasa videolist
    if videolist is not None:
        videos = [v for v in videos if v[0] in videolist]
        #print(videos)
    # Lista de redes
    if net is None:
        nets = ('yolov5m6', 'yolov5x6', 'yolov5x')
    else:
        nets = (net,)

    for net in nets:
        print(f"Processing network: '{net}'")
        for video, ext, legendPlace, legendTextSize, recortar in videos:
            print(f"Processing video: '{video}'")
            for method in methods:
                #print(f"Processing method: '{method}'")
                
                name = f"{video}_{net}" 
                frame_file = glob.glob(os.path.join(output_folder,"*","first_frame",f"*{video}*.png"), recursive = True)[0]
                frame = cv2.imread(frame_file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                labels = {"0%":"0_noise", "5%":"5_noise", "10%":"10_noise", "20%":"20_noise"}
                labels = {"20%":"20_noise","10%":"10_noise","5%":"5_noise","0%":"0_noise"  }
                #label_colors = {"0%": "#1f77b4", "5%": "#9acd82", "10%": "#d62728", "20%": "#8c1762"}
                label_colors = {"0%": "#BBCC33", "5%": "#F9D576", "10%": "#FD9A44", "20%": "#B2182B"}
                label_colors = {"20%": "#B2182B", "10%": "#FD9A44", "5%": "#F9D576","0%": "#BBCC33"}
                marker = "o"
                legend_info = []
                #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                
                fig, ax = plt.subplots(figsize=(8,6))
                ax.imshow(frame)
                for i, label in enumerate(labels.keys()):
                    file = sorted(glob.glob(os.path.join(base_dir,output_folder,labels[label],"**",f"{video}_{net}_{method}_centroids.csv"), recursive = True))[0]
                    df = pd.read_csv(file)
                    #color = colors[i % len(colors)]
                    color = label_colors[label]

                    ax.scatter(df["x"],df["y"],color = color,marker=marker,s=200, alpha=0.7, edgecolor="black", linewidth=0.5)

                    legend_info.append((color, label))

                ax.axis("off")

                frame_out = os.path.join(output_folder, "centroids_noise_addition",f"{name}_{method}_centroids.png")
                os.makedirs(os.path.dirname(frame_out), exist_ok=True)
                fig.savefig(frame_out, dpi=300, bbox_inches="tight")
                plt.close(fig)

        fig_leg, ax_leg = plt.subplots(figsize=(2.2, 2.2))

        for color, label in reversed(legend_info):
            ax_leg.scatter([], [], 
                        color=color, 
                        marker=marker, 
                        s=160,        
                        edgecolor="none",
                        label=label)

        legend = ax_leg.legend(
            loc="center",
            frameon=False,       
            fontsize=12,
            handletextpad=0.8,
            labelspacing=1.2,
            borderpad=0.5
        )

        ax_leg.axis("off")

        ax_leg.set_title(
            "Added noise percentage",
            fontsize=14,
            pad=12
        )

        legend_out = os.path.join(
            output_folder,
            "centroids_noise_addition",
            f"{name}_{method}_centroids_legend.png"
        )

        fig_leg.savefig(legend_out, dpi=300, bbox_inches="tight")
        plt.close(fig_leg)


def merge_metrics_to_csv(input_dir:str):
    # Listar todos los CSV de métricas (archivos que empiezan con 'tabla_')
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and f.startswith('tabla_')]

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV de métricas en '{input_dir}'.")

    # Diccionario para almacenar los DataFrames por métrica
    metric_dfs = {}

    for csv_file in csv_files:
        metrica = csv_file.replace('tabla_', '').replace('.csv', '').replace('_', ' ')
        df = pd.read_csv(os.path.join(input_dir, csv_file), index_col=0)
        # Eliminar filas Mean y Std si existen
        df = df.drop([r for r in ['Mean', 'Std'] if r in df.index], errors='ignore')
        metric_dfs[metrica] = df

    # Tomar videos y métodos desde cualquiera de los DataFrames
    any_df = next(iter(metric_dfs.values()))
    videos = any_df.index.tolist()
    metodos = any_df.columns.tolist()

    # Construir filas
    rows = []
    for video in videos:
        for metodo in metodos:
            fila = {
                'video': video,
                'network': 'yolov5x6',
                'Clustering method': metodo
            }
            for metrica in [
                'Number of clusters', 'MSE', 'DBI',
                'Silhouette score', 'Calinski-Harabasz', 'Runtime (ms)'
            ]:
                if metrica in metric_dfs and metodo in metric_dfs[metrica].columns and video in metric_dfs[metrica].index:
                    fila[metrica] = metric_dfs[metrica].at[video, metodo]
                else:
                    fila[metrica] = None
            rows.append(fila)

    # Crear DataFrame final
    df_merged = pd.DataFrame(rows)

    # Guardar CSV final
    path = os.path.join(os.path.dirname(input_dir),"perfMeasures.csv")
    df_merged.to_csv(path, index=False)
    print(f"Merged CSV as {path}")


def prepare_dfs(df_0, df_5, df_10, df_20):

    dfs = {
        "0": df_0.copy(),
        "5": df_5.copy(),
        "10": df_10.copy(),
        "20": df_20.copy()
    }

    for k in dfs:
        dfs[k].columns = dfs[k].columns.str.strip()

    return dfs


def format_result(result, video_order):

    result = result.reset_index()

    result["video"] = pd.Categorical(
        result["video"],
        categories=video_order,
        ordered=True
    )

    result = result.sort_values(["video", "diff"])
    result = result.set_index(["video", "diff"])

    result = result.rename(index={
        "videoDiagonal1": "CARLA1",
        "videoDiagonal2": "CARLA2",
        "videoHorizontal": "CARLA3"
    }, level="video")

    return result


def get_delta_csv(df_0, df_5, df_10, df_20, output_folder):

    dfs = prepare_dfs(df_0, df_5, df_10, df_20)

    metrics = [
        "MSE",
        "DBI",
        "Silhouette score",
        "Calinski-Harabasz",
        "Runtime (ms)"
    ]

    methods_col = "Clustering method"
    video_col = "video"

    for k in dfs:
        dfs[k] = dfs[k].set_index([video_col, methods_col])

    diffs = {
        "5-0": dfs["5"][metrics] - dfs["0"][metrics],
        "10-0": dfs["10"][metrics] - dfs["0"][metrics],
        "20-0": dfs["20"][metrics] - dfs["0"][metrics],
    }

    video_order = [
        'videoDiagonal1','videoDiagonal2','videoHorizontal','Highway',
        'Seq1_SK_1','Seq1_SK_4','Seq2_SK_1','Seq2_SK_4','Seq3_SK_1','Seq3_SK_4',
        'Hadsundvej-1','Hadsundvej-2','Hasserisvej-1','Hasserisvej-2','Hasserisvej-3',
        'Hjorringvej-2','Ostre-3'
    ]

    for metric in metrics:

        rows = []

        for label, df_diff in diffs.items():

            tmp = df_diff[[metric]].reset_index()

            pivot = tmp.pivot(
                index=video_col,
                columns=methods_col,
                values=metric
            )

            pivot["diff"] = label
            pivot = pivot.reset_index().set_index([video_col, "diff"])

            rows.append(pivot)

        result = pd.concat(rows)

        result = result.reset_index()
        result["diff_order"] = result["diff"].str.split("-").str[0].astype(int)

        result = result.sort_values([video_col, "diff_order"])
        result = result.drop(columns="diff_order")
        result = result.set_index([video_col, "diff"])

        result = result[['Geometric Heuristic','K-means (Elbow)','K-means (Silhouette)','Mean Shift','DBSCAN']]
        result.rename(columns={'Geometric Heuristic':'Quadrant-based'}, inplace=True)

        result = format_result(result, video_order)

        fname = metric.replace(" ", "_").replace("(", "").replace(")", "")
        out_dir = os.path.join(output_folder, "delta","abs")
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"{fname}.csv")

        result.to_csv(filename)

        print(f"Saved: {filename}")


def plot_delta_heatmaps_subplots(delta_folder, output_folder, scale_range=None, fontsize=12):

    os.makedirs(output_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(delta_folder) if f.endswith(".csv")]

    maximize_metrics = ["Silhouette score", "Calinski-Harabasz"]

    comparisons = ["5-0", "10-0", "20-0"]
    labels = {"5-0": "$\Delta_{5}$", "10-0": "$\Delta_{10}$", "20-0": "$\Delta_{20}$"}

    for file in csv_files:

        metric_name = file.replace(".csv", "")
        path = os.path.join(delta_folder, file)

        df = pd.read_csv(path)

        methods = [c for c in df.columns if c not in ["video", "diff"]]

        invert = (
            metric_name in maximize_metrics
            or metric_name.replace("_", " ") in maximize_metrics
        )

        cmap = "seismic_r" if invert else "seismic"

        n_videos = df["video"].nunique()
        fig_height = max(4, n_videos * 0.4)

        fig, axes = plt.subplots(1, 3, figsize=(15, fig_height), sharey=True)

        for i, comp in enumerate(comparisons):

            ax = axes[i]

            df_comp = df[df["diff"] == comp].copy()
            heatmap_data = df_comp.set_index("video")[methods]

            # definir escala
            if scale_range is not None:
                vmin, vmax = scale_range
            else:
                vmax = np.abs(heatmap_data.values).max()
                vmin = -vmax

                #if "Silhouette" in metric_name:
                #    vmin, vmax = -0.25, 0.25

            # detectar valores fuera de rango
            extend = "neither"
            if heatmap_data.values.max() > vmax and heatmap_data.values.min() < vmin:
                extend = "both"
            elif heatmap_data.values.max() > vmax:
                extend = "max"
            elif heatmap_data.values.min() < vmin:
                extend = "min"

            hm = sns.heatmap(
                heatmap_data,
                cmap=cmap,
                center=0,
                annot=False,
                linewidths=0.5,
                ax=ax,
                cbar=(i == 2),
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"extend": extend} if i == 2 else None
            )
            if i == 2:
                cbar = hm.collections[0].colorbar
                cbar.ax.tick_params(labelsize=fontsize*0.6)
            # título subplot
            ax.set_title(labels[comp], fontsize=fontsize)

            # tamaño ticks
            ax.tick_params(axis="y", labelsize=fontsize * 0.6, rotation=0)
            ax.tick_params(axis="x", labelsize=fontsize * 0.7, rotation=90)

            if i > 0:
                ax.set_ylabel("")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        path_fig = os.path.join(
            output_folder,
            "heatmaps_subplots",
            os.path.basename(delta_folder),
        )

        os.makedirs(path_fig, exist_ok=True)

        plt.savefig(os.path.join(path_fig, metric_name + "_heatmap.png"), dpi=300)
        plt.savefig(
            os.path.join(path_fig, metric_name + "_heatmap.eps"),
            format="eps",
        )

        plt.close()

def plot_delta_heatmaps_grid(delta_folder, output_folder, scale_range=None, fontsize=12):

    metrics_order = [
        "MSE",
        "DBI",
        "Calinski-Harabasz",
        "Silhouette_score",
    ]

    maximize_metrics = ["Silhouette_score", "Calinski-Harabasz"]

    comparisons = ["5-0", "10-0", "20-0"]
    labels = {"5-0": "$\Delta_{5}$", "10-0": "$\Delta_{10}$", "20-0": "$\Delta_{20}$"}

    metric_data = {}

    for metric in metrics_order:
        file = metric + ".csv"
        path = os.path.join(delta_folder, file)
        df = pd.read_csv(path)
        methods = [c for c in df.columns if c not in ["video", "diff"]]
        metric_data[metric] = (df, methods)

    n_videos = df["video"].nunique()
    fig_height = max(6, n_videos * 0.35)

    fig, axes = plt.subplots(
        2,
        6,
        figsize=(24, fig_height),
        sharey=False,  # IMPORTANTE
    )

    metric_positions = {
        "MSE": (0, 0),
        "DBI": (0, 3),
        "Calinski-Harabasz": (1, 0),
        "Silhouette_score": (1, 3),
    }

    for metric, (row, col_start) in metric_positions.items():

        df, methods = metric_data[metric]

        invert = metric in maximize_metrics

        cmap = "seismic_r" if invert else "seismic"

        if scale_range is not None:
            vmin, vmax = scale_range
        else:
            all_vals = []
            for comp in comparisons:
                vals = df[df["diff"] == comp][methods].values
                all_vals.append(vals)

            all_vals = np.concatenate(all_vals)
            vmax = np.abs(all_vals).max()
            vmin = -vmax

            if "Silhouette" in metric:
                vmin, vmax = -0.25, 0.25

        for i, comp in enumerate(comparisons):

            ax = axes[row, col_start + i]

            df_comp = df[df["diff"] == comp]
            heatmap_data = df_comp.set_index("video")[methods]

            extend = "neither"
            if heatmap_data.values.max() > vmax and heatmap_data.values.min() < vmin:
                extend = "both"
            elif heatmap_data.values.max() > vmax:
                extend = "max"
            elif heatmap_data.values.min() < vmin:
                extend = "min"

            sns.heatmap(
                heatmap_data,
                cmap=cmap,
                center=0,
                annot=False,
                linewidths=0.5,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cbar=(col_start + i == 5),
                cbar_kws={"extend": extend} if (col_start + i == 5) else None,
                yticklabels=heatmap_data.index if (col_start + i) in [0,3] and i == 0 else False,
            )

            ax.set_ylabel("")

            # títulos 5% 10% 20%
            if row == 0:
                ax.set_title(labels[comp], fontsize=fontsize)

            # vídeos solo en primera columna de cada bloque
            if (col_start + i) in [0,3] and i == 0:
                ax.tick_params(axis="y", labelsize=fontsize * 0.3, rotation=0, length=0)
            else:
                ax.tick_params(axis="y", labelleft=False)

            # métodos solo abajo
            if row == 0:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis="x", rotation=90, labelsize=fontsize * 0.8)

        center_ax = axes[row, col_start + 1]
        center_ax.text(
            0.5,
            1.25,
            metric.replace("_", " "),
            transform=center_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=fontsize * 1.2,
            fontweight="bold",
        )

    fig.tight_layout()

    path_fig = os.path.join(
        output_folder,
        "heatmaps_subplots",
        os.path.basename(delta_folder),
    )

    os.makedirs(path_fig, exist_ok=True)

    fig.savefig(os.path.join(path_fig, "heatmap_grid.png"), dpi=300)
    fig.savefig(os.path.join(path_fig, "heatmap_grid.eps"), format="eps")

    plt.close(fig)
def plot_delta_heatmaps_column(delta_folder, output_folder, scale_range=None, fontsize=12):

    metrics_order = [
        "MSE",
        "DBI",
        "Calinski-Harabasz",
        "Silhouette_score",
    ]

    maximize_metrics = ["Silhouette score", "Calinski-Harabasz"]

    comparisons = ["5-0", "10-0", "20-0"]
    labels = {"5-0": "$\Delta_{5}$", "10-0": "$\Delta_{10}$", "20-0": "$\Delta_{20}$"}

    metric_data = {}

    for metric in metrics_order:
        file = metric + ".csv"
        path = os.path.join(delta_folder, file)
        df = pd.read_csv(path)
        methods = [c for c in df.columns if c not in ["video", "diff"]]
        metric_data[metric] = (df, methods)

    n_videos = df["video"].nunique()
    fig_height = max(12, n_videos * 0.5)

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(9, fig_height),
        sharey=False,  # CLAVE
    )

    for row, metric in enumerate(metrics_order):

        df, methods = metric_data[metric]

        invert = (
            metric in maximize_metrics
            or metric.replace("_", " ") in maximize_metrics
        )

        cmap = "seismic_r" if invert else "seismic"

        if scale_range is not None:
            vmin, vmax = scale_range
        else:
            all_vals = []
            for comp in comparisons:
                vals = df[df["diff"] == comp][methods].values
                all_vals.append(vals)

            all_vals = np.concatenate(all_vals)
            vmax = np.abs(all_vals).max()
            vmin = -vmax

            if "Silhouette" in metric:
                vmin, vmax = -0.25, 0.25

        for col, comp in enumerate(comparisons):

            ax = axes[row, col]

            df_comp = df[df["diff"] == comp]
            heatmap_data = df_comp.set_index("video")[methods]

            extend = "neither"
            if heatmap_data.values.max() > vmax and heatmap_data.values.min() < vmin:
                extend = "both"
            elif heatmap_data.values.max() > vmax:
                extend = "max"
            elif heatmap_data.values.min() < vmin:
                extend = "min"

            sns.heatmap(
                heatmap_data,
                cmap=cmap,
                center=0,
                annot=False,
                linewidths=0.5,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cbar=(col == 2),
                cbar_kws={"extend": extend} if col == 2 else None,
                yticklabels=heatmap_data.index if col == 0 else False,
            )

            ax.set_ylabel("")

            if row == 0:
                ax.set_title(labels[comp], fontsize=fontsize, pad=6)

            if col == 0:
                ax.tick_params(axis="y", labelsize=fontsize * 0.55, length=0)
            else:
                ax.tick_params(axis="y", labelleft=False)

            if row == len(metrics_order) - 1:
                ax.tick_params(axis="x", rotation=90, labelsize=fontsize * 0.8)
            else:
                ax.set_xticklabels([])

        axes[row, 0].set_ylabel(
            metric.replace("_", " "),
            fontsize=fontsize * 0.8,
            weight="bold",
            rotation=90,
            labelpad=10,
        )

    fig.tight_layout(pad=0.8)

    path_fig = os.path.join(
        output_folder,
        "heatmaps_subplots",
        os.path.basename(delta_folder),
    )
    os.makedirs(path_fig, exist_ok=True)

    fig.savefig(os.path.join(path_fig, "heatmap_column.png"), dpi=300)
    fig.savefig(os.path.join(path_fig, "heatmap_column.eps"), format="eps")
    plt.close(fig)

if __name__=='__main__':
    # Apply the YOLOv5 model to detect, norfair method to track saving tracked points and generates 5-10-20 added noise
    videolist =  ['videoDiagonal1','videoDiagonal2', 'videoHorizontal', 'Highway', 'Seq1_SK_1', 'Seq1_SK_4', 'Seq2_SK_1', 'Seq2_SK_4', 'Seq3_SK_1', 'Seq3_SK_4', 'Hadsundvej-1', 'Hadsundvej-2', 'Hasserisvej-1', 'Hasserisvej-2', 'Hasserisvej-3','Hjorringvej-2', 'Ostre-3']
    save_initial_final_points_by_detection_tracking('output_rereview','yolov5x6',videolist, 'yolov5demo.py')
    
    output_folder = "output_rereview"

    # generates clusters with the proposed clustering methods and computes the chosen performance metrics
    scriptEv(net='yolov5x6', videolist= videolist, output_folder = output_folder) 
    dfResults = pd.read_csv(os.path.join(base_dir, output_folder+"/perfMeasures/perfMeasures.csv"))
    split_by_metric(dfResults, os.path.join(base_dir, output_folder, "perfMeasures","by_metric"))
    # get performance metrics for noisy versions
    for n in [5,10,20]:
        output_folder = f"output_rereview/{n}_noise"
        scriptEv(net='yolov5x6', videolist= videolist, output_folder = output_folder)
        dfResults = pd.read_csv(os.path.join(base_dir, output_folder+"/perfMeasures/perfMeasures.csv"))
        split_by_metric(dfResults, os.path.join(base_dir, output_folder, "perfMeasures","by_metric"))

    paint_centroids(net='yolov5x6', videolist=videolist, output_folder = os.path.join(base_dir,"output_rereview"))
    
    merge_metrics_to_csv("/output_rereview/0_noise/perfMeasures/by_metric")

    df_0 = pd.read_csv(os.path.join(base_dir, output_folder,"0_noise","perfMeasures","perfMeasures.csv"))
    df_5 = pd.read_csv(os.path.join(base_dir, output_folder,"5_noise","perfMeasures","perfMeasures.csv"))
    df_10 = pd.read_csv(os.path.join(base_dir, output_folder,"10_noise","perfMeasures","perfMeasures.csv"))
    df_20 = pd.read_csv(os.path.join(base_dir, output_folder,"20_noise","perfMeasures","perfMeasures.csv"))

    # get difference between non-noise and noise added version (abs and percent)
    get_delta_csv(df_0,df_5,df_10,df_20, os.path.join(base_dir, output_folder))
    get_delta_pct_csv(df_0,df_5,df_10,df_20, os.path.join(base_dir, output_folder))
