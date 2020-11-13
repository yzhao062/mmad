import numpy as np
from vae import VAE
from pyod.utils.data import generate_data, evaluate_print
from pyod.utils.utility import standardizer

if __name__ == "__main__":
    # contamination = 0.1  # percentage of outliers
    # n_train = 20000  # number of training points
    # n_test = 2000  # number of testing points


    X_image = np.load('train_image_embedding.npy')
    X_text = np.load('word2vec.npy')
    
    X = np.concatenate([X_image, X_text], axis=1)
    n_features = X.shape[1]  # number of features
    
    X_transformed = standardizer(X)
    # # train VAE detector (Beta-VAE)
    clf_name = 'VAE'
    clf = VAE(epochs=50, latent_dim=128, gamma=1, capacity=0)
    clf.fit(X_transformed)

    # # get the prediction labels and outlier scores of the training data
    # y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf.decision_scores_  # raw outlier scores

    # # get the prediction on the test data
    # y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    # y_test_scores = clf.decision_function(X_test)  # outlier scores

    # # evaluate and print the results
    # print("\nOn Training Data:")
    # evaluate_print(clf_name, y_train, y_train_scores)
    # print("\nOn Test Data:")
    # evaluate_print(clf_name, y_test, y_test_scores)


    latent_dim = clf.encoder_.predict(X_transformed)[2]
    np.save('vae_joint_representation1.npy', latent_dim)
