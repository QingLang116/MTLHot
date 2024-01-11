import numpy as np
from sklearn.cluster import KMeans
from keras.models import Model, load_model
from keras.optimizers import adam_v2
from ClusteringLayer import *


def get_model_layers(model_file, num_layers):
    model = load_model(model_file)
    # define new model that cuts off the last several layers
    newmodel = Model(inputs=model.input, outputs=model.layers[num_layers].output)
    # agian, need to specify these parameters, but they aren't used since we don't retrain the model
    opt = adam_v2.Adam()
    newmodel.compile(optimizer=opt, loss=None)
    return newmodel


# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def DEC(mod_to_layer, n_clusters, X_train, maxiterV, update_interval, tol, batch_size):

    # -----------------------------定义聚类模型----------------------------------------#
    print("model building:----------------")
    # 为编码层增加一个聚类层
    clustering_layer = ClusteringLayer(n_clusters=n_clusters, name='clustering')(mod_to_layer.output)
    model = Model(inputs=mod_to_layer.input, outputs=clustering_layer)
    # 固定encoder层的模型参数，不参与模型训练
    for layer in model.layers[0:len(mod_to_layer.layers)]:
        layer.trainable = False  # 前两个神经层固定不训练，只训练聚类层
    model.compile(optimizer='adam', loss='kld')

    # ----------------------------------聚类 -------------------------------------#
    print("Initialize cluster centers:------------------")
    np.random.seed(10)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(mod_to_layer.predict(X_train))
    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    print("deep clustering:--------------")
    loss = 0
    index = 0
    maxiter = maxiterV
    update_interval = update_interval
    index_array = np.arange(X_train.shape[0])
    tol = tol  # tolerance threshold to stop training，
    batch_size = batch_size

    # Training
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(X_train, verbose=0)
            p = target_distribution(q)
            y_pred = q.argmax(1)

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, X_train.shape[0])]
        loss = model.train_on_batch(x=X_train[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= X_train.shape[0] else 0

    return model
