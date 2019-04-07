import pickle
import numpy as np
import gensim
from sklearn import preprocessing
import pandas as pd
import os

X_train = pickle.load(open('x-train', 'rb'))
y_train = pickle.load(open('y-train', 'rb'))
X_test = pickle.load(open('x-test', 'rb'))
y_test = pickle.load(open('y-test', 'rb'))


def average_embedding(tokens, word2vec, na_vector=None):

    """ Embeds a title with the average representation of its tokens.

    Returns the mean vector representation of the tokens representations. When no token is in the Word2Vec model, it
    can be provided a vector to use instead (for example the mean vector representation of the train set titles).

    @param tokens: List of tokens to embed.
    @param word2vec: Word2Vec model.
    @param na_vector: Vector representation to use when no token is in the Word2Vec model.
    @return: A vector representation for the token list.
    """

    vectors = list()

    for token in tokens:
        if token in word2vec:
            vectors.append(word2vec[token])

    if len(vectors) == 0 and na_vector is not None:
        vectors.append(na_vector)

    return np.mean(np.array(vectors), axis=0)


documents = X_train["video_title_tokenized"]
word2vec = gensim.models.Word2Vec(
    documents,
    size=25,
    window=20,
    min_count=1,
    workers=2
)
word2vec.train(documents, total_examples=len(documents), epochs=30)

# Export it:
pickle.dump(word2vec, open("word2vec", "wb"))


titles_embeddings = X_train["video_title_tokenized"].apply(average_embedding, word2vec=word2vec)
train_set = pd.concat(
    [
        X_train[["video_views", "video_likes", "video_dislikes", "video_comments"]],
        titles_embeddings.apply(pd.Series)
    ], axis=1)
# Add the label column:
train_set["label"] = y_train
# Drop rows with missing values:
train_set = train_set.dropna()

# Compute the average vector representation on the train set, and export it:
mean_title_embedding = titles_embeddings.dropna().mean(axis=0)
pickle.dump(mean_title_embedding, open("mean-title-embedding", "wb"))

# For the test set use the mean title embedding computed on the train set:
titles_embeddings = X_test["video_title_tokenized"].apply(average_embedding, word2vec=word2vec, na_vector=mean_title_embedding)
test_set = pd.concat(
    [
        X_test[["video_views", "video_likes", "video_dislikes", "video_comments"]],
        titles_embeddings.apply(pd.Series)
    ], axis=1)
test_set["label"] = y_test

print(train_set.shape, test_set.shape)

# Compute the logarithm of the video metadata (likes, dislikes, comments, views)
train_set[["video_views", "video_likes", "video_dislikes", "video_comments"]] = train_set[["video_views", "video_likes", "video_dislikes", "video_comments"]].apply(np.log)
test_set[["video_views", "video_likes", "video_dislikes", "video_comments"]] = test_set[["video_views", "video_likes", "video_dislikes", "video_comments"]].apply(np.log)
# Replace any -Inf value with 0:
train_set = train_set.replace(-np.inf, 0)
test_set = test_set.replace(-np.inf, 0)

# Remove the label columns:
train_labels = train_set["label"]
test_labels = test_set["label"]
train_set = train_set.drop(columns=["label"])
test_set = test_set.drop(columns=["label"])

# Export the mean values of the metadata in the train set:
pickle.dump(train_set["video_views"].mean(), open("mean-log-video-views", "wb"))
pickle.dump(train_set["video_likes"].mean(), open("mean-log-video-likes", "wb"))
pickle.dump(train_set["video_dislikes"].mean(), open("mean-log-video-dislikes", "wb"))
pickle.dump(train_set["video_comments"].mean(), open("mean-log-video-comments", "wb"))

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(train_set)
train_set = pd.DataFrame(min_max_scaler.transform(train_set), columns=train_set.columns)
test_set = pd.DataFrame(min_max_scaler.transform(test_set), columns=test_set.columns)
# Export it:
pickle.dump(min_max_scaler, open("min-max-scaler", "wb"))


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

svm_params = [
    { "C": np.linspace(1, 25, 10), "gamma": np.linspace(1, 5, 10) },
]


print(test_set)
exists = os.path.isfile('svm')
if exists:
    grid_search_cv = pickle.load(open('svm', 'rb'))
    predictions = grid_search_cv.predict(test_set)
    print("Best SVM with:")
    print("Performance on the test set (%d samples):" % len(test_set))
    print("\tAccuracy Score:", accuracy_score(test_labels, predictions))
    print("\tArea under ROC curve:", roc_auc_score(test_labels, predictions))
    print("\tClassification report (on the test set):")
    print(classification_report(test_labels, predictions))


else:
    grid_search_cv = GridSearchCV(estimator=SVC(kernel="rbf", probability=True), param_grid=svm_params, n_jobs=2, scoring="f1", verbose=3)
    grid_search_cv.fit(train_set, train_labels)
    predictions = grid_search_cv.predict(test_set)

    # Export the best estimator:
    pickle.dump(grid_search_cv.best_estimator_, open("svm", "wb"))
    print("Best SVM with:")
    print("\tC:", grid_search_cv.best_params_["C"])
    print("\tgamma:", grid_search_cv.best_params_["gamma"])
    print("\tBest Score (F1):", grid_search_cv.best_score_)
    print("Performance on the test set (%d samples):" % len(test_set))
    print("\tAccuracy Score:", accuracy_score(test_labels, predictions))
    print("\tArea under ROC curve:", roc_auc_score(test_labels, predictions))
    print("\tClassification report (on the test set):")
    print(classification_report(test_labels, predictions))



print(grid_search_cv.predict_proba(test_set))



