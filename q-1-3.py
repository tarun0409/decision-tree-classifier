#!/usr/bin/env python
# coding: utf-8

# <html>
#   <body>
#       <h2>Entropy, Gini and Misclassification Rate</h2>
#       <h3>Q1.3. Contrast the effectiveness of Misclassification rate, Gini, Entropy as impurity measures in terms of precision, recall and accuracy.</h3>
#       <p>Here we train the data set to make a decision tree on numerical and categorical data and contrast the performance based on the impurity function we use.</p>
#       <p>The impurity functions examined here are:</p>
#       <ol>
#         <li>Entropy</li>
#         <li>Gini</li>
#         <li>Misclassification Rage</li>
#       </ol>
#    </body>
# </html>

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

data = pd.read_csv('train.csv')


# <html>
#   <body>
#     <p>The original set is split into 80% train data 20% test data</p>
#   </body>
# </html>

# In[2]:


X_train, X_test, y_train, y_test = train_test_split(
    data,
    data.left,
    test_size=0.2,
    random_state=0)


# <html>
#   <body>
#     <p>While there are different ways to deal with numerical data when building a decision tree, here we split it into ranges based on their contribution to highest entropy.</p>
#     <p>After deciding the ranges, we convert each numerical data point into categorical data point based on the range to which they belong to.</p>
#   </body>
# </html>

# In[3]:


map_range_sl = lambda x: '<=0.17' if x<=0.17 else ('0.18 to 0.29' if x <= 0.29 else ('0.30 to 0.39' if x<=0.39 else ('0.40 to 0.46' if x<=0.46 else ('0.47 to 0.53' if x<=0.53 else ('0.54 to 0.6' if x<=0.6 else ('0.61 to 0.67' if x<=0.67 else ('0.68 to 0.74' if x<=0.74 else ('0.75 to 0.8' if x<=0.8 else ('0.81 to 0.87' if x<=0.87 else ('0.88 to 0.94' if x<=0.94 else '>0.94'))))))))))
X_train.satisfaction_level = X_train.satisfaction_level.apply(map_range_sl)
X_test.satisfaction_level = X_test.satisfaction_level.apply(map_range_sl)

map_range_le = lambda x: '<=0.44' if x<=0.44 else ('0.45 to 0.51' if x <= 0.51 else ('0.52 to 0.58' if x<=0.58 else ('0.59 to 0.65' if x<=0.65 else ('0.66 to 0.71' if x<=0.71 else ('0.72 to 0.78' if x<=0.78 else ('0.79 to 0.86' if x<=0.86 else ('0.87 to 0.93' if x<=0.93 else '>0.93'))))))) 
X_train.last_evaluation = X_train.last_evaluation.apply(map_range_le)
X_test.last_evaluation = X_test.last_evaluation.apply(map_range_le)

map_range_np = lambda x: '<=3' if x<=3 else '>3'
X_train.number_project = X_train.number_project.apply(map_range_np)
X_test.number_project = X_test.number_project.apply(map_range_np)

map_range_amh = lambda x: '<=173' if x<=173 else ('174 to 229' if x<=229 else '>229')
X_train.average_montly_hours = X_train.average_montly_hours.apply(map_range_amh)
X_test.average_montly_hours = X_test.average_montly_hours.apply(map_range_amh)

map_range_ts = lambda x : '<=3' if x <=3 else ('>3 and <=6' if x<=6 else '>6')
X_train.time_spend_company = X_train.time_spend_company.apply(map_range_ts)
X_test.time_spend_company = X_test.time_spend_company.apply(map_range_ts)


# <html>
#   <body>
#     <p>The following function creates a node with the details:</p>
#     <ul>
#       <li><b>value</b>: The actual value the node is supposed to represent. It can be a column name or a result value</li>
#       <li><b>is_leaf</b>: A boolean value indicating whether the current node is a leaf</li>
#       <li><b>children</b>: This is a python dictionary where the key represents branching condition and value represents the child node that the branch leads to</li>
#     </ul>
#   </body>
# </html>

# In[4]:


def create_tree_node(value, is_leaf):
    new_node = dict()
    new_node['value'] = value
    new_node['is_leaf'] = is_leaf
    new_node['children'] = dict()
    return new_node


# <html>
#   <body>
#     <p>The following function computes the impurity. For this question, we explre three impurity functions</p>
#     <ul>
#       <li>Entropy
#         <br>
#         $$E = -(qlogq + (1-q)log(1-q))$$
#         <br>
#       </li>
#       <li>Gini
#         <br>
#         $$E = 2q(1-q)$$
#         <br>
#       </li>
#       <li>Misclassification Rate
#         <br>
#         $$E = min(q,1-q)$$
#         <br>
#       </li>
#     </ul>
#   </body>
# </html>
# 

# In[5]:


def compute_impurity(pos_ratio,neg_ratio,impurity_function):
    if impurity_function == 'entropy':
        if pos_ratio == 0 or pos_ratio == 1:
            return 0
        else:
            return -((pos_ratio*np.log(pos_ratio)) + (neg_ratio*np.log(neg_ratio)))
    elif impurity_function == 'gini_index':
        return (2*pos_ratio*neg_ratio)
    elif impurity_function == 'misclassification_rate':
        return min(pos_ratio,neg_ratio)
    return 0


# <html>
#   <body>
#     <p>We now introduce a python class to create training and predict functions as member functions. We also store the decision tree in a member variable of the same class.</p>
#     <p>The following are the member functions of the class:</p>
#     <ul>
#       <li><b>train</b>: Will take the training data and create the decision tree</li>
#       <li><b>predict</b>: Will take the input data and predict the class to which each row belongs to</li>
#       <li><b>get_model</b>: Will return the decision tree object</li>
#       <li><b>compute_accuracy</b>: Will check predicted value with the actual value and compute the accuracy</li>
#       <li><b>get_precision_recall_f1score</b>: Will compute precision, recall and F1 Score. Formula given later</li>
#     </ul>
#   </body>
# </html>

# In[6]:


class DecisionTree:
    
    decision_tree = None
    
    def get_model(self):
        return self.decision_tree
    
    def get_row_result(self,row,tree_node):
        if tree_node['is_leaf']:
            return tree_node['value']
        
        key = tree_node['value']
        val = row[key]
        if val not in tree_node['children']:
            return 0
        else:
            return self.get_row_result(row,tree_node['children'][val])
    
    def predict(self,X):
        y_pred = list()
        for index,row in X.iterrows():
            result = self.get_row_result(row,self.decision_tree)
            y_pred.append(result)
        df = pd.DataFrame({'Y_predicted': y_pred})
        return df
    
    def compute_accuracy(self,Y_actual,Y_predict):
        ya = Y_actual.values.tolist()
        yp = Y_predict.values.tolist()
        l = len(ya)
        count = 0
        for i in range(0,l):
            if int(ya[i])==int(yp[i]):
                count+=1
        return float(count)/float(l)
    
    def get_precision_recall_f1score(self,Y_actual,Y_predict):
        ya = Y_actual.values.tolist()
        yp = Y_predict.values.tolist()
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        l = len(ya)
        for i in range(0,l):
            if int(ya[i])==0 and int(yp[i])==0:
                #true negative
                tn += 1
            elif int(ya[i])==0 and int(yp[i])==1:
                #false positive
                fp += 1
            elif int(ya[i])==1 and int(yp[i])==0:
                #false negative
                fn += 1
            else:
                #true positive
                tp += 1
        precision = float(tp)/(float(tp)+float(fp))
        recall = float(tp)/(float(tp)+float(fn))
        f1_score = 2.0/((1.0/recall)+(1.0/precision))
        return (precision,recall,f1_score)
    
    
    def train(self, curr_data,curr_parent_object,curr_condition, exclude_cols, impurity_function):
        curr_parent_str = curr_parent_object['value']
        if curr_parent_str not in exclude_cols:
            exclude_cols.append(curr_parent_str)
        m = len(curr_data)
        G = dict()
        for col in curr_data:
            if col in exclude_cols:
                continue
            I = 0.0
            col_pos = 0
            col_neg = 0
            int_info = 0
            for categ in curr_data[col].unique():
                pos = 0
                neg = 0
                if 1 in curr_data.groupby([col])['left'].value_counts()[categ]: 
                    pos = curr_data.groupby([col])['left'].value_counts()[categ][1]
                if 0 in curr_data.groupby([col])['left'].value_counts()[categ]:
                    neg = curr_data.groupby([col])['left'].value_counts()[categ][0]
                col_pos += pos
                col_neg += neg
                pos_ratio = float(pos)/float(pos+neg)
                neg_ratio = float(neg)/float(pos+neg)
                impurity = compute_impurity(pos_ratio,neg_ratio, impurity_function)
                Si_S = float(pos+neg)/float(m) 
                I += Si_S*impurity
                int_info += Si_S*np.log(Si_S)
            col_pos_ratio = float(col_pos)/float(col_pos+col_neg)
            col_neg_ratio = float(col_neg)/float(col_pos+col_neg)
            E = compute_impurity(col_pos_ratio,col_neg_ratio, impurity_function)
            G[col] = ((E-I)/(-1*int_info))
    
        if not G:
            if curr_parent_str == 'dummy_parent':
                return self.decision_tree
            for categ in curr_data[col].unique():
                if 0 in curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition] and 1 in curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition]:
                    if curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition][0] > curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition][1]:
                        new_node = create_tree_node('0',True)
                        curr_parent_object['children'][curr_condition] = new_node
                    else:
                        new_node = create_tree_node('1',True)
                        curr_parent_object['children'][curr_condition] = new_node
                elif 0 in curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition]:
                    new_node = create_tree_node('0',True)
                    curr_parent_object['children'][curr_condition] = new_node
                elif 1 in curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition]:
                    new_node = create_tree_node('1',True)
                    curr_parent_object['children'][curr_condition] = new_node
                else:
                    new_node = create_tree_node('0',True)
                    curr_parent_object['children'][curr_condition] = new_node
            return self.decision_tree
    
        max_col = max(G, key=G.get)
        
        new_node = create_tree_node(max_col,False)
        if self.decision_tree is None:
            self.decision_tree = new_node
        else:
            curr_parent_object['children'][curr_condition] = new_node

        temp_exclude_cols = exclude_cols[:]
        for cond in curr_data[max_col].unique():
            self.train(curr_data.loc[curr_data[max_col]==cond].copy() , new_node, cond,temp_exclude_cols, impurity_function)
    
        return self.decision_tree


# <html>
#   <body>
#     <p>The decision tree is built using the following algorithm: </p>
#     <ol>
#       <li>Input data</li>
#       <li>Compute impurities (entropy) of all columns</li>
#       <li>Compute Average entropy of all columns with the following formula</li>
#       $$I(S,A) = \sum_{i}^{ }\frac{|{S_{i}}^{ }|}{|S|}\cdot E({S_{i}}^{ })$$
#       <li>Compute Gain of columns using the following formula</li>
#       $$G(S,A) = E(S) - I(S,A)$$
#       <li>If no more Gain is left to compute, declare leaf node and fill corresponding result value</li>
#       <li>Find the column with maximum gain and make it a node</li>
#       <li>For each category of column with maximum Gain, repeat from step 2</li>
#     </ol>
#   </body>
# </html>

# In[7]:


dummy_node = dict()
dummy_node['value'] = 'dummy_parent'

entropy_dt = DecisionTree()
gini_dt = DecisionTree()
mr_dt = DecisionTree()

entropy_model = entropy_dt.train(X_train.copy(),dummy_node,'dummy_condition',['left'],'entropy')
gini_model = gini_dt.train(X_train.copy(),dummy_node,'dummy_condition',['left'],'gini_index')
mr_model = mr_dt.train(X_train.copy(),dummy_node,'dummy_condition',['left'],'misclassification_rate')

entropy_train_predict = entropy_dt.predict(X_train)
gini_train_predict = gini_dt.predict(X_train)
mr_train_predict = mr_dt.predict(X_train)

entropy_acc = entropy_dt.compute_accuracy(X_train['left'],entropy_train_predict['Y_predicted'])
gini_acc = gini_dt.compute_accuracy(X_train['left'],gini_train_predict['Y_predicted'])
mr_acc = mr_dt.compute_accuracy(X_train['left'],mr_train_predict['Y_predicted'])

print 'Train Entropy model Accuracy : '+str(entropy_acc*100)
print 'Train Gini index model Accuracy : '+str(gini_acc*100)
print 'Train Misclassification Rate index model Accuracy : '+str(mr_acc*100)


# In[8]:


entropy_test_predict = entropy_dt.predict(X_test)
gini_test_predict = gini_dt.predict(X_test)
mr_test_predict = mr_dt.predict(X_test)

entropy_acc = entropy_dt.compute_accuracy(X_test['left'],entropy_test_predict['Y_predicted'])
gini_acc = gini_dt.compute_accuracy(X_test['left'],gini_test_predict['Y_predicted'])
mr_acc = mr_dt.compute_accuracy(X_test['left'],mr_test_predict['Y_predicted'])

print 'Test Entropy model Accuracy : '+str(entropy_acc*100)
print 'Test Gini index model Accuracy : '+str(gini_acc*100)
print 'Test Misclassification Rate index model Accuracy : '+str(mr_acc*100)


# <html>
#   <body>
#     <p>Precision, recall and F1 score are computed using the following formula: </p>
#     <br>
#     $$Precision = \frac{True Positive}{True Positive + False Positive}$$
#     <br>
#     $$Recall = \frac{True Positive}{True Positive + False Negative}$$
#     <br>
#     $$F1 Score = \frac{2}{\frac{1}{Recall} + \frac{1}{Precision}}$$
#     <br>
#   </body>
# </html>

# In[9]:


entropy_precision,entropy_recall,entropy_f1_score = entropy_dt.get_precision_recall_f1score(X_train['left'],entropy_train_predict['Y_predicted'])
print 'Train Entropy Precision : '+str(entropy_precision)
print 'Train Entropy Recall : '+str(entropy_recall)
print 'Train Entropy F1 Score : '+str(entropy_f1_score)
print
print
gini_precision,gini_recall,gini_f1_score = gini_dt.get_precision_recall_f1score(X_train['left'],gini_train_predict['Y_predicted'])
print 'Train Gini Index Precision : '+str(gini_precision)
print 'Train Gini Index Recall : '+str(gini_recall)
print 'Train Gini Index F1 Score : '+str(gini_f1_score)
print
print
mr_precision,mr_recall,mr_f1_score = mr_dt.get_precision_recall_f1score(X_train['left'],mr_train_predict['Y_predicted'])
print 'Train Gini Index Precision : '+str(mr_precision)
print 'Train Gini Index Recall : '+str(mr_recall)
print 'Train Gini Index F1 Score : '+str(mr_f1_score)
print
print


# In[10]:


entropy_precision,entropy_recall,entropy_f1_score = entropy_dt.get_precision_recall_f1score(X_test['left'],entropy_test_predict['Y_predicted'])
print 'Test Entropy Precision : '+str(entropy_precision)
print 'Test Entropy Recall : '+str(entropy_recall)
print 'Test Entropy F1 Score : '+str(entropy_f1_score)
print
print
gini_precision,gini_recall,gini_f1_score = gini_dt.get_precision_recall_f1score(X_test['left'],gini_test_predict['Y_predicted'])
print 'Test Gini Index Precision : '+str(gini_precision)
print 'Test Gini Index Recall : '+str(gini_recall)
print 'Test Gini Index F1 Score : '+str(gini_f1_score)
print
print
mr_precision,mr_recall,mr_f1_score = mr_dt.get_precision_recall_f1score(X_test['left'],mr_test_predict['Y_predicted'])
print 'Test Gini Index Precision : '+str(mr_precision)
print 'Test Gini Index Recall : '+str(mr_recall)
print 'Test Gini Index F1 Score : '+str(mr_f1_score)
print
print


# In[ ]:




