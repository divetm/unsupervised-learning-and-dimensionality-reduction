from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import scale
from sklearn import mixture
from sklearn.metrics import accuracy_score

from scipy import linalg

import itertools

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


def plot_digits(digits_data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(digits_data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
    plt.show()


def choose_k_means():
    n_clusters = np.arange(1, 100, 5)
    plus_plus_models = [KMeans(init='k-means++', n_clusters=n, n_init=10)
                        for n in n_clusters]
    random_models = [KMeans(init='random', n_clusters=n, n_init=10)
                   for n in n_clusters]
    plus_plus_times = []
    for model in plus_plus_models:
        t0 = time()
        model.fit(data)
        plus_plus_times.append(time()-t0)
    random_times = []
    for model in random_models:
        t0 = time()
        model.fit(data)
        random_times.append(time()-t0)
    plt.plot(n_clusters, plus_plus_times, label='K-means ++')
    plt.plot(n_clusters, random_times, label='Random')
    plt.legend(loc='best')
    plt.xlabel('n_clusters')
    plt.ylabel('Computation time (s)')
    plt.show()
    plt.figure()
    plus_plus_v_meas = [metrics.v_measure_score(labels, model.fit(data).predict(data)) for model in plus_plus_models]
    random_v_meas = [metrics.v_measure_score(labels, model.fit(data).predict(data)) for model in random_models]
    plt.plot(n_clusters, plus_plus_v_meas, label='K-means ++')
    plt.plot(n_clusters, random_v_meas, label='Random')
    plt.legend(loc='best')
    plt.xlabel('n_clusters')
    plt.ylabel('V-measure score')
    plt.show()


def choose_em():
    n_components = np.arange(1, 100, 5)
    spherical_models = [mixture.GaussianMixture(n, covariance_type='spherical', max_iter=20, random_state=0)
              for n in n_components]
    diag_models = [mixture.GaussianMixture(n, covariance_type='diag', max_iter=20, random_state=0)
              for n in n_components]
    tied_models = [mixture.GaussianMixture(n, covariance_type='tied', max_iter=20, random_state=0)
              for n in n_components]
    full_models = [mixture.GaussianMixture(n, covariance_type='full', max_iter=20, random_state=0)
              for n in n_components]
    spherical_bics = [model.fit(data).bic(data) for model in spherical_models]
    diag_bics = [model.fit(data).bic(data) for model in diag_models]
    tied_bics = [model.fit(data).bic(data) for model in tied_models]
    full_bics = [model.fit(data).bic(data) for model in full_models]
    plt.plot(n_components, spherical_bics, label='Spherical')
    plt.plot(n_components, diag_bics, label='Diagonal')
    plt.plot(n_components, tied_bics, label='Tied')
    plt.plot(n_components, full_bics, label='Full')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('BIC score')
    plt.show()
    plt.figure()
    spherical_v_meas = [metrics.v_measure_score(labels, model.fit(data).predict(data)) for model in spherical_models]
    diag_v_meas = [metrics.v_measure_score(labels, model.fit(data).predict(data)) for model in diag_models]
    tied_v_meas = [metrics.v_measure_score(labels, model.fit(data).predict(data)) for model in tied_models]
    full_v_meas = [metrics.v_measure_score(labels, model.fit(data).predict(data)) for model in full_models]
    plt.plot(n_components, spherical_v_meas, label='Spherical')
    plt.plot(n_components, diag_v_meas, label='Diagonal')
    plt.plot(n_components, tied_v_meas, label='Tied')
    plt.plot(n_components, full_v_meas, label='Full')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('V-measure score')
    plt.show()
    plt.figure()
    spherical_times = []
    for model in spherical_models:
        t0 = time()
        model.fit(data)
        spherical_times.append(time() - t0)
    diag_times = []
    for model in diag_models:
        t0 = time()
        model.fit(data)
        diag_times.append(time() - t0)
    tied_times = []
    for model in tied_models:
        t0 = time()
        model.fit(data)
        tied_times.append(time() - t0)
    full_times = []
    for model in full_models:
        t0 = time()
        model.fit(data)
        full_times.append(time() - t0)
    plt.plot(n_components, spherical_times, label='Spherical')
    plt.plot(n_components, diag_times, label='Diagonal')
    plt.plot(n_components, tied_times, label='Tied')
    plt.plot(n_components, full_times, label='Full')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('Computation time (s)')
    plt.show()
    return full_bics


def choose_pca():
    variance_explained_ratios = []
    for i in range(1, 65):
        pca = PCA(n_components=i).fit(data)
        variance_explained_ratios.append(pca.explained_variance_ratio_)
    variance_csv = ""
    for vars in variance_explained_ratios:
        for var in vars:
            variance_csv += str(var) + "; "
        variance_csv += "//"
    return variance_csv


def compare_dimensionality_reduction_times():
    plt.figure()
    n_components = [i for i in range(1, 65)]
    PCA_algorithms = [PCA(n_components=n) for n in n_components]
    ICA_algorithms = [FastICA(n_components=n) for n in n_components]
    RCA_algorithms = [SparseRandomProjection(n_components=n) for n in n_components]
    LDA_algorithms = [LinearDiscriminantAnalysis(n_components=n) for n in n_components]
    PCA_times = []
    for algo in PCA_algorithms:
        t0 = time()
        algo.fit_transform(data)
        PCA_times.append(time() - t0)
    ICA_times = []
    for algo in ICA_algorithms:
        t0 = time()
        algo.fit_transform(data)
        ICA_times.append(time() - t0)
    RCA_times = []
    for algo in RCA_algorithms:
        t0 = time()
        algo.fit_transform(data)
        RCA_times.append(time() - t0)
    LDA_times = []
    for algo in LDA_algorithms:
        t0 = time()
        algo.fit_transform(data, labels)
        LDA_times.append(time() - t0)
    plt.plot(n_components, PCA_times, label='PCA')
    plt.plot(n_components, ICA_times, label='ICA')
    plt.plot(n_components, RCA_times, label='RCA')
    plt.plot(n_components, LDA_times, label='LDA')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('Computation time (s)')
    plt.show()


def lda_scores_depending_on_n_components():
    plt.figure()
    n_components = [i for i in range(1, 10)]
    LDA_fitted_algorithms = [LinearDiscriminantAnalysis(n_components=n).fit(data, labels) for n in n_components]
    scores = []
    for algo in LDA_fitted_algorithms:
        scores.append(algo.score(data, labels))
    plt.plot(n_components, scores, label='LDA')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.show()


def k_means_and_pca_2D_graph():
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    clusters = kmeans.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def k_means_and_pca_optimal_accuracy():
    reduced_data = PCA(n_components=31).fit_transform(data)
    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    clusters = kmeans.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def k_means_and_ica_2D_graph():
    reduced_data = FastICA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (ICA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    clusters = kmeans.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def k_means_and_ica_optimal_accuracy():
    reduced_data = FastICA(n_components=31).fit_transform(data)
    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    clusters = kmeans.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def k_means_and_rca_2D_graph():
    reduced_data = SparseRandomProjection(n_components=2).fit_transform(data)
    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (RCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    clusters = kmeans.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def k_means_and_rca_optimal_accuracy():
    reduced_data = SparseRandomProjection(n_components=31).fit_transform(data)
    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    clusters = kmeans.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def k_means_and_lda_2D_graph():
    reduced_data = LinearDiscriminantAnalysis(n_components=2).fit_transform(data, labels)
    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (LDA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    clusters = kmeans.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def k_means_and_lda_optimal_accuracy():
    reduced_data = LinearDiscriminantAnalysis(n_components=9).fit_transform(data, labels)
    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    clusters = kmeans.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def plot_em_results(X, Y_, means, covariances, index, title):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                  'darkorange'])
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()


def em_and_pca_2D_graph():
    reduced_data = PCA(n_components=2).fit_transform(data)
    gm = mixture.GaussianMixture(n_digits, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(reduced_data)
    plot_em_results(reduced_data, gm.predict(reduced_data), gm.means_, gm.covariances_, 0,
                    'EM clustering on the tumor dataset (PCA-reduced data)')
    clusters = gm.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def em_and_pca_optimal_accuracy():
    reduced_data = PCA(n_components=31).fit_transform(data)
    gm = mixture.GaussianMixture(n_digits, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(reduced_data)
    clusters = gm.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def em_and_ica_2D_graph():
    reduced_data = FastICA(n_components=2).fit_transform(data)
    gm = mixture.GaussianMixture(n_digits, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(reduced_data)
    plot_em_results(reduced_data, gm.predict(reduced_data), gm.means_, gm.covariances_, 0,
                    'EM clustering on the tumor dataset (ICA-reduced data)')
    clusters = gm.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def em_and_ica_optimal_accuracy():
    reduced_data = FastICA(n_components=31).fit_transform(data)
    gm = mixture.GaussianMixture(n_digits, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(reduced_data)
    clusters = gm.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def em_and_rca_2D_graph():
    reduced_data = SparseRandomProjection(n_components=2).fit_transform(data)
    gm = mixture.GaussianMixture(n_digits, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(reduced_data)
    plot_em_results(reduced_data, gm.predict(reduced_data), gm.means_, gm.covariances_, 0,
                    'EM clustering on the tumor dataset (RCA-reduced data)')
    clusters = gm.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def em_and_rca_optimal_accuracy():
    reduced_data = SparseRandomProjection(n_components=31).fit_transform(data)
    gm = mixture.GaussianMixture(n_digits, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(reduced_data)
    clusters = gm.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def em_and_lda_2D_graph():
    reduced_data = LinearDiscriminantAnalysis(n_components=2).fit_transform(data, labels)
    gm = mixture.GaussianMixture(n_digits, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(reduced_data)
    plot_em_results(reduced_data, gm.predict(reduced_data), gm.means_, gm.covariances_, 0,
                    'EM clustering on the tumor dataset (LDA-reduced data)')
    clusters = gm.predict(reduced_data)
    print(accuracy_score(labels, clusters))


def em_and_lda_optimal_accuracy():
    reduced_data = LinearDiscriminantAnalysis(n_components=9).fit_transform(data, labels)
    gm = mixture.GaussianMixture(n_digits, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(reduced_data)
    clusters = gm.predict(reduced_data)
    print(accuracy_score(labels, clusters))


# choose_k_means()
# bics = choose_em()
# print(choose_pca())
# compare_dimensionality_reduction_times()
# lda_scores_depending_on_n_components()
# k_means_and_pca_2D_graph()
# k_means_and_ica_2D_graph()
# k_means_and_rca_2D_graph()
# k_means_and_lda_2D_graph()
# em_and_pca_2D_graph()
# em_and_ica_2D_graph()
# em_and_rca_2D_graph()
# em_and_lda_2D_graph()
# TODO: change values of the reduction algorithms to the ones given by the choose functions in the optimal accuracy
#  functions and measure scores of the resulting clusterings (when n_clusters = 2 for tumor problem and 10 for digit
#  problem) => compare to measure how better the reduction is
# k_means_and_pca_optimal_accuracy()
# k_means_and_ica_optimal_accuracy()
# k_means_and_rca_optimal_accuracy()
# k_means_and_lda_optimal_accuracy()
# em_and_pca_optimal_accuracy()
# em_and_ica_optimal_accuracy()
# em_and_rca_optimal_accuracy()
# em_and_lda_optimal_accuracy()
