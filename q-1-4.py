#!/usr/bin/env python
# coding: utf-8

# <html>
#   <body>
#       <h2>Data Visualisation</h2>
#       <h3>Q1.4. Visualise training data on a 2-dimensional plot taking one feature (attribute) on one axis and other feature on another axis. Take two suitable features to visualise decision tree boundary</h3>
#       <p>Here we visualize the data using the following two features</p>
#       <ol>
#         <li>satisfaction_level</li>
#         <li>last_evaluation</li>
#       </ol>
#   </body>
# </html>

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')


# In[4]:


df1 = data.loc[data['left']==0]
df2 = data.loc[data['left']==1]

ax1 = df1.plot.scatter(
    x='satisfaction_level',
    y='last_evaluation',
    c='green',
    label = 'y=0'
)
ax2 = df2.plot.scatter(
    x='satisfaction_level',
    y='last_evaluation',
    c='brown',
    label='y=1',
    ax = ax1
)

plt.show()
# <html>
#   <body>
#     <p>As we can see above the taking just features satisfaction_level and last_evaluation, the employees seem to have left the company for the following conditions</p>
#     <ul>
#       <li>satisfaction_level > 0.09 and satisfaction_level < 0.2 and last_evaluation > 0.75 and last_evaluation < 0.98</li>
#       <li>satisfaction_level > 0.35 and satisfaction_level < 0.45 and last_evaluation > 0.45 and last_evaluation < 0.55</li>
#       <li>satisfaction_level > 0.7 and satisfaction_level < 0.9 and last_evaluation > 0.75</li>
#     </ul>
#     <p>The employees seem to have NOT left the company for the following conditions</p>
#     <ul>
#       <li>satisfaction_level > 0.5 and last_evaluation > 0.45</li>
#       <li>satisfaction_level > 0.1 and satisfaction_level< 0.25 and last_evaluation >0.5 </li>
#     </ul>
#   </body>
# </html>
# 

# In[ ]:





# In[ ]:




