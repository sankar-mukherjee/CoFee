
#Different sklearn clustering analysis techniques and plotting and dimension reduction. dataprep.py must be run before.

X_train, y_train = get_clusterdata(data,REAL_POS_FEAT+REAL_ACO_FEAT,'simple',WORKING_DIR + 'cluster_classifier.png')  

#full features
#X_train, y_train = get_clusterdata(data,REAL_POS_FEAT+REAL_ACO_FEAT,'baseFun0.65',WORKING_DIR+'cluster_classifier.png')

reduced_data = X_train
labels = y_train
######################################dimension reduction #################################

#################### PCA varience cpture
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
reduced_data = pca.fit(X_train).transform(X_train)

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
      
#################### #####################   LDA
from sklearn.lda import LDA
lda = LDA(n_components=3)
scale = lda.fit(X_train,y_train)
reduced_data = scale.transform(X_train)
####################################### ICA ####################################
from sklearn.decomposition import FastICA
# Compute ICA
ica = FastICA(n_components=3)
reduced_data = ica.fit_transform(X_train)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
#######################################################plot 3D pca with 3 components #################
import mpl_toolkits.mplot3d.axes3d as p3

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot3D(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],'.')
#########################################clustering ################################
############ K means
le = preprocessing.LabelEncoder()
le.fit(y_train)
le.classes_
ori_label = le.transform(y_train)

kmeans = KMeans(init='k-means++',n_clusters=11, n_init=1)
kmeans.fit(X_train)
reduced_data = kmeans.predict(X_train)
pred_label = kmeans.labels_

bench_k_means(kmeans,name="k-means++",data=X_train)

pred = reduced_data - ori_label
np.count_nonzero(pred)

kmeans = KMeans(init='k-means++',n_init=10)
kmeans.fit(reduced_data)

bench_k_means(KMeans(init='k-means++', n_init=10, n_clusters= 2),
              name="k-means++", data=reduced_data)

bench_k_means(KMeans(init='random', n_clusters=10, n_init=10),
              name="random", data=reduced_data)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

## plot
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.show()

############################ Compute DBSCAN
from sklearn.cluster import DBSCAN

bench_k_means(DBSCAN(eps=0.01, min_samples=10), name="DBSCAN", data=reduced_data)

db = DBSCAN(eps=0.01, min_samples=10).fit(reduced_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = reduced_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = reduced_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

################################### Compute Affinity Propagation###################################
from sklearn.cluster import AffinityPropagation

af = AffinityPropagation(preference=-50).fit(reduced_data)
cluster_centers_indices = af.cluster_centers_indices_
labels_true = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(reduced_data, labels, metric='sqeuclidean'))

# Plot result
plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = reduced_data[cluster_centers_indices[k]]
    plt.plot(reduced_data[class_members, 0], reduced_data[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in reduced_data[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

################################# Compute clustering with MeanShift###############################
from sklearn.cluster import MeanShift, estimate_bandwidth
# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(reduced_data, quantile=0.1, n_samples=100)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(reduced_data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

bench_k_means(ms, name="MeanShift", data=reduced_data)

# Plot result
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(reduced_data[my_members, 0], reduced_data[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

################################# AgglomerativeClustering #############
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
# Create a graph capturing local connectivity. Larger number of neighbors
# will give more homogeneous clusters to the cost of computation
# time. A very large number of neighbors gives more evenly distributed
# cluster sizes, but may not impose the local manifold structure of
# the data
knn_graph = kneighbors_graph(reduced_data, 30, mode='distance')

for connectivity in (None, knn_graph):
    for n_clusters in (30, 3):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average', 'complete', 'ward')):
            plt.subplot(1, 3, index + 1)
            model = AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(reduced_data)
            elapsed_time = time.time() - t0
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=model.labels_,
                        cmap=plt.cm.spectral)
            plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
                      fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')

            plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                left=0, right=1)
            plt.suptitle('n_cluster=%i, connectivity=%r' %
                         (n_clusters, connectivity is not None), size=17)


plt.show()

################################### Birch #######################################
# Compute clustering with Birch with and without the final clustering step
# and plot.
from sklearn.cluster import Birch

# Use all colors that matplotlib provides by default.
colors_ = cycle(colors.cnames.keys())

birch_models = [Birch(threshold=1.7, n_clusters=None),
                Birch(threshold=1.7, n_clusters=100)]
final_step = ['without global clustering', 'with global clustering']

fig = plt.figure(figsize=(12, 4))
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)

for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
    birch_model.fit(reduced_data)

    # Plot result
    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = np.unique(labels).size
    print("n_clusters : %d" % n_clusters)

    ax = fig.add_subplot(1, 3, ind + 1)
    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k
        ax.plot(reduced_data[mask, 0], reduced_data[mask, 1], 'w',
                markerfacecolor=col, marker='.')
        if birch_model.n_clusters is None:
            ax.plot(this_centroid[0], this_centroid[1], '+', markerfacecolor=col,
                    markeredgecolor='k', markersize=5)
    ax.set_ylim([-25, 25])
    ax.set_xlim([-25, 25])
    ax.set_autoscaley_on(False)
    ax.set_title('Birch %s' % info)











