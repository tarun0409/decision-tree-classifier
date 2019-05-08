#!/usr/bin/env python
# coding: utf-8

# <html>
#   <body>
#       <h2>Performance w.r.t Depth and Number of Nodes in Tree </h2>
#       <h3>Q1.5. Plot a graph of training and validation error with respect to depth of your decision tree. Also plot the training and validation error with respect to number of nodes in the decision tree.</h3>
#       <p>Here we train the data set and analyse the performance w.r.t the following properties of the tree:</p>
#       <ul>
#           <li>Depth of tree</li>
#           <li>Number of Nodes in tree</li>
#       </ul>
#       <p>We plot line graphs based on the data received from above</p>
#   </body>
# </html>

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

data = pd.read_csv('train.csv')
data.head()


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
X_train.head()


# <html>
#   <body>
#     <p>The following function creates a node with the details:</p>
#     <ul>
#       <li><b>value</b>: The actual value the node is supposed to represent. It can be a column name or a result value</li>
#       <li><b>is_leaf</b>: A boolean value indicating whether the current node is a leaf</li>
#       <li><b>children</b>: This is a python dictionary where the key represents branching condition and value represents the child node that the branch leads to</li>
#       <li><b>height</b>: This is the height of node in the tree. Tree with one node has height 1</li>
#       <li><b>number</b>: Serial number of the node create. First node has number 1</li>
#     </ul>
#   </body>
# </html>

# In[4]:


def create_tree_node(value, is_leaf,number,height):
    new_node = dict()
    new_node['value'] = value
    new_node['is_leaf'] = is_leaf
    new_node['children'] = dict()
    new_node['height'] = height
    new_node['number'] = number
    return new_node


# <html>
#   <body>
#     <p>The following function computes the impurity. For this question, we restrict ourselves to computing the impurity using <b>entropy</b> formula</p>
#     <p>The following is the formula for computing the entropy</p>
#     <br>
#     $$E = -(qlogq + (1-q)log(1-q))$$
#     <br>
#   </body>
# </html>

# In[5]:


def compute_impurity(pos_ratio,neg_ratio):
    if pos_ratio == 0 or pos_ratio == 1:
        return 0
    else:
        return -((pos_ratio*np.log(pos_ratio)) + (neg_ratio*np.log(neg_ratio)))


# In[6]:


class DecisionTree:
    
    decision_tree = None
    
    def get_model(self):
        return self.decision_tree
    
    def get_row_result(self,row,tree_node,criteria,max_items):
        if tree_node['is_leaf']:
            return tree_node['value']
        
        if criteria == 'height':
            if tree_node['height'] >= max_items:
                pc = tree_node['pos_count']
                nc = tree_node['neg_count']
                if pc > nc:
                    return 1
                else:
                    return 0
        elif criteria == 'number':
            if tree_node['number'] >= max_items:
                pc = tree_node['pos_count']
                nc = tree_node['neg_count']
                if pc > nc:
                    return 1
                else:
                    return 0
        
        key = tree_node['value']
        val = row[key]
        if val not in tree_node['children']:
            return 0
        else:
            return self.get_row_result(row,tree_node['children'][val],criteria,max_items)
    
    def predict(self,X,criteria,max_items):
        y_pred = list()
        for index,row in X.iterrows():
            result = self.get_row_result(row,self.decision_tree, criteria, max_items)
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
    
    def train(self, curr_data,curr_parent_object,curr_condition, exclude_cols,number_of_nodes,height):
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
                impurity = compute_impurity(pos_ratio,neg_ratio)
                Si_S = float(pos+neg)/float(m) 
                I += Si_S*impurity
            int_info += Si_S*np.log(Si_S)
            col_pos_ratio = float(col_pos)/float(col_pos+col_neg)
            col_neg_ratio = float(col_neg)/float(col_pos+col_neg)
            E = compute_impurity(col_pos_ratio,col_neg_ratio)
            G[col] = ((E-I)/(-1*int_info))
    
        if not G:
            if curr_parent_str == 'dummy_parent':
                return self.decision_tree
            for categ in curr_data[col].unique():
                if 0 in curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition] and 1 in curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition]:
                    if curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition][0] > curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition][1]:
                        number_of_nodes += 1
                        new_node = create_tree_node('0',True, number_of_nodes, height)
                        curr_parent_object['children'][curr_condition] = new_node
                    else:
                        number_of_nodes += 1
                        new_node = create_tree_node('1',True, number_of_nodes, height)
                        curr_parent_object['children'][curr_condition] = new_node
                elif 0 in curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition]:
                    number_of_nodes += 1
                    new_node = create_tree_node('0',True, number_of_nodes, height)
                    curr_parent_object['children'][curr_condition] = new_node
                elif 1 in curr_data.groupby([curr_parent_str])['left'].value_counts()[curr_condition]:
                    number_of_nodes += 1
                    new_node = create_tree_node('1',True, number_of_nodes, height)
                    curr_parent_object['children'][curr_condition] = new_node
                else:
                    number_of_nodes += 1
                    new_node = create_tree_node('0',True, number_of_nodes, height)
                    curr_parent_object['children'][curr_condition] = new_node
            return (self.decision_tree,number_of_nodes)
    
        max_col = max(G, key=G.get)
    
        number_of_nodes += 1
        new_node = create_tree_node(max_col,False, number_of_nodes, height)
            
        if self.decision_tree is None:
            self.decision_tree = new_node
        else:
            curr_parent_object['children'][curr_condition] = new_node
    
        if new_node['is_leaf']:
            return (self.decision_tree,number_of_nodes)
    
    
        temp_exclude_cols = exclude_cols[:]
        for cond in curr_data[max_col].unique():
            if 0 in curr_data.groupby([max_col])['left'].value_counts()[cond]:
                neg_count = curr_data.groupby([max_col])['left'].value_counts()[cond][0]
            else:
                neg_count = 0
            if 1 in curr_data.groupby([max_col])['left'].value_counts()[cond]:
                pos_count = curr_data.groupby([max_col])['left'].value_counts()[cond][1]
            else:
                pos_count = 0
            new_node['pos_count'] = pos_count
            new_node['neg_count'] = neg_count
            self.decision_tree,number_of_nodes = self.train(curr_data.loc[curr_data[max_col]==cond].copy() , new_node, cond,temp_exclude_cols,number_of_nodes,(height+1))
    
        return (self.decision_tree,number_of_nodes)


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

dt = DecisionTree()
model = dt.train(X_train.copy(),dummy_node,'dummy_condition',['left'], 0, 1)


# In[8]:


heights = list()
train_accuracies = list()
test_accuracies = list()
for h in range(1,11):
    heights.append(h)
    left_predict = dt.predict(X_train, 'height', h)
    acc = dt.compute_accuracy(X_train['left'],left_predict['Y_predicted'])
    train_accuracies.append(acc)
for h in range(1,11):
    left_predict = dt.predict(X_test, 'height', h)
    acc = dt.compute_accuracy(X_test['left'],left_predict['Y_predicted'])
    test_accuracies.append(acc)
df = pd.DataFrame({'Train Performance': train_accuracies,'Test Performance': test_accuracies}, index=heights)
lines = df.plot.line()


# In[10]:


numbers = list()
train_accuracies = list()
test_accuracies = list()
for h in range(1,17108,100):
    numbers.append(h)
    left_predict = dt.predict(X_train, 'number', h)
    acc = dt.compute_accuracy(X_train['left'],left_predict['Y_predicted'])
    train_accuracies.append(acc)
for h in range(1,17108,100):
    left_predict = dt.predict(X_test, 'number', h)
    acc = dt.compute_accuracy(X_test['left'],left_predict['Y_predicted'])
    test_accuracies.append(acc)
df = pd.DataFrame({'Train Performance': train_accuracies,'Test Performance': test_accuracies}, index=numbers)
lines = df.plot.line()

plt.show()
# In[ ]:




