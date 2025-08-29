
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# Load Data
def load_json(path, labeled=True):
    data, labels, ids = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text = " ".join(map(str, obj["text"]))
            data.append(text)
            ids.append(obj["id"])
            if labeled:
                labels.append(obj["label"])
    return data, labels if labeled else None, ids

X1, y1, _ = load_json("dataset/domain1_train_data.json", labeled=True)
X2, y2, _ = load_json("dataset/domain2_train_data.json", labeled=True)
X_test, _, test_ids = load_json("dataset/test_data.json", labeled=False)

# Domain Classifier（Logistic)
X_domain = X1 + X2
y_domain = [0] * len(X1) + [1] * len(X2)
vec_dom = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=5000)
X_domain_vec = vec_dom.fit_transform(X_domain)
X_test_dom_vec = vec_dom.transform(X_test)

clf_domain = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_domain.fit(X_domain_vec, y_domain)
domain_pred = clf_domain.predict(X_test_dom_vec)

# Model Training for Domain1, Domain2
vec_word = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=10000)
vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=5000)
vec_word.fit(X1 + X2)
vec_char.fit(X1 + X2)

X1_vec = hstack([vec_word.transform(X1), vec_char.transform(X1)])
X2_vec = hstack([vec_word.transform(X2), vec_char.transform(X2)])
X_test_vec = hstack([vec_word.transform(X_test), vec_char.transform(X_test)])

model_d1 = LogisticRegression(max_iter=1000, class_weight='balanced')
model_d1.fit(X1_vec, y1)

model_d2 = LogisticRegression(max_iter=1000, class_weight='balanced')
model_d2.fit(X2_vec, y2)

# Final Prediction
final_labels = []
for i in range(len(X_test)):
    xi = X_test_vec[i]
    if domain_pred[i] == 0:
        pred = model_d1.predict(xi)
    else:
        pred = model_d2.predict(xi)
    final_labels.append(pred[0])


# Data Save
df_output = pd.DataFrame({
    "id": test_ids,
    "class": final_labels
})
df_output.to_csv("prediction.csv", index=False)
print("Saved: prediction.csv")


