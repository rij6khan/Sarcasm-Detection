#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
valid_df = pd.read_csv("valid.csv")
train_df = pd.read_csv("train.csv")
train_df.head()
train_df.info()
train_df['label'].value_counts()


# In[2]:


train_df['text_length'] = train_df['text'].astype(str).apply(len)
train_df['text_length'].describe()


# In[3]:


train_df.nlargest(5, 'text_length')[['text_length','text']]


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),    # bigrams important
        min_df=2,
        max_df=0.95,
        stop_words=None
    )),
    ("clf", LinearSVC())
])


# #### This Pipeline has two main components:
# 
# TfidfVectorizer: converts raw text into numerical feature vectors (TF-IDF), using both unigrams and bigrams.
# 
# LinearSVC: a linear Support Vector Machine classifier which takes the TF-IDF vectors as input and predicts class labels (0 = not sarcastic, 1 = sarcastic).
# 
# The Pipeline wraps them so you can call .fit() and .predict() directly on raw text.
# 
# #### 定义一个 baseline 模型 
# 
# 1. TfidfVectorizer(...)：
# ·把句子变成数字特征（TF-IDF 向量）
# ·用 unigram + bigram（(1,2)）来抓住“短语级讽刺特征”
# 
# 2. LinearSVC( )：·一个线性 SVM 分类器
# ·输入 TF-IDF 向量，输出 0 / 1（非讽刺 / 讽刺）
# 
# Pipeline 把这两步串起来，让你可以 fit 和 predict 时自动完成“文本→向量→分类”。

# In[5]:


baseline.fit(train_df["text"], train_df["label"])


# In[6]:


from sklearn.metrics import classification_report, confusion_matrix

valid_pred = baseline.predict(valid_df["text"])

print(classification_report(valid_df["label"], valid_pred))
confusion_matrix(valid_df["label"], valid_pred)


# In[7]:


from sklearn.model_selection import GridSearchCV

params = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [1,2,3],
    "clf__C": [0.1, 1, 3, 5 ]
}

grid = GridSearchCV(
    baseline,
    param_grid=params,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1
)

grid.fit(train_df["text"], train_df["label"])
grid.best_params_, grid.best_score_

# show all the result of grid search
results = pd.DataFrame(grid.cv_results_)

results = results[[
    "param_tfidf__ngram_range",
    "param_tfidf__min_df",
    "param_clf__C",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]]

results = results.sort_values("rank_test_score")

results


# This does:
# 
# 3-fold cross-validation inside train_df
# 
# Tries different combinations of:
# 
# ngram_range (unigrams vs unigrams+bigrams)
# 
# min_df (how rare a term must be to be dropped)
# 
# C (regularization strength of the SVM)
# 
# grid.best_params_ tells you the best hyperparameter combination,
# grid.best_score_ gives the corresponding average CV F1-score.
# 
# Also extracted all combinations with grid.cv_results_ and sorted them for analysis.
# 
# 
# 用 train_df 自己内部 做 3-fold 交叉验证（cv=3）
# 
# 尝试不同组合：
# 
# ngram: 只用 unigram，还是用 uni+bigram
# 
# min_df: 要不要过滤掉特别稀有词
# 
# C: SVM 的“强度”（复杂度）
# 
# grid.best_params_：哪一组参数最好
# grid.best_score_：这组参数在 cross-validation 上的平均 F1
# 
# 你还把所有组合的结果拿出来排了个表（cv_results_）

# In[8]:


full_df = pd.concat([train_df, valid_df], ignore_index=True)

final_model = grid.best_estimator_
final_model.fit(full_df["text"], full_df["label"])


# In[9]:


import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

test_df = pd.read_csv("test.csv")

test_pred = final_model.predict(test_df["text"])

print(classification_report(test_df["label"], test_pred))
print(confusion_matrix(test_df["label"], test_pred))


# train.csv
#    ↓
# GridSearchCV（inner validation）→ find the best hyperparameter
#    ↓
# use the best hyperparameter + train.csv + valid.csv → train final model
#    ↓
# use test.csv → get the result

# In[ ]:





# In[ ]:




