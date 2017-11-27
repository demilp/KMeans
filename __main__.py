import numpy as np
import matplotlib.pyplot as plt
import KMeans

def run():
    points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([2, 0])),
                        (np.random.randn(50, 2) * 0.25 + np.array([-1.5, 0.5])),
                        (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))
    K = 3
    kmeans = KMeans.KMeans()
    kmeans.fit(points, K, 10)
    #plt.scatter(points[:, 0], points[:, 1])
    c = kmeans.centroids

    t = np.random.randn(1000, 2) * 1
    cid = kmeans.predict(t)
    for k in np.arange(K):
        w = t[cid==k]
        plt.scatter(w[:, 0], w[:, 1])

    plt.scatter(c[:, 0], c[:, 1])

    plt.show()




if __name__ == '__main__':
    run()