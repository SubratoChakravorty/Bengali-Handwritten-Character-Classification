#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


filepaths = ['history_convnet.csv', 'history_resnet.csv', 'history_invresnet.csv', 
             'history_mobnetv1_2.csv','history_mobnetv2_2.csv','history_dense.csv']

# Model names
titles = ['Convnet', 'Resnet', 'InvertedResnet', 'Mobilenet-v1','Mobilenet-v2','Dense' ]

df = pd.DataFrame()

# Plot parameters
params = ['dense_3_loss','dense_4_loss','dense_5_loss',
          'val_dense_3_loss','val_dense_4_loss','val_dense_5_loss','dense_3_accuracy','dense_4_accuracy','dense_5_accuracy','val_dense_3_accuracy','val_dense_4_accuracy','val_dense_5_accuracy']
actual_params = ['grapheme_root_loss','vowel_diacritic_loss','consonant_diacritic_loss',
                'val_grapheme_root_loss','val_vowel_diacritic_loss','val_consonant_diacritic_loss',
                 'grapheme_root_accuracy','vowel_diacritic_accuracy','consonant_diacritic_accuracy','val_grapheme_root_accuracy','val_vowel_diacritic_accuracy','val_consonant_diacritic_accuracy']
# Create a dataframe with all the parameters for all models
for i, filepath in enumerate(filepaths):
    history_df = pd.read_csv(filepath)
    title = titles[i]
    if i<5:
        df[title+'_grapheme_root_loss'] = history_df['dense_3_loss']
        df[title+'_vowel_diacritic_loss'] = history_df['dense_4_loss']
        df[title+'_consonant_diacritic_loss'] = history_df['dense_5_loss']

        df[title+'_val_grapheme_root_loss'] = history_df['val_dense_3_loss']
        df[title+'_val_vowel_diacritic_loss'] = history_df['val_dense_4_loss']
        df[title+'_val_consonant_diacritic_loss'] = history_df['val_dense_5_loss']
        
        df[title+'_grapheme_root_accuracy'] = history_df['dense_3_accuracy']
        df[title+'_vowel_diacritic_accuracy'] = history_df['dense_4_accuracy']
        df[title+'_consonant_diacritic_accuracy'] = history_df['dense_5_accuracy']

        df[title+'_val_grapheme_root_accuracy'] = history_df['val_dense_3_accuracy']
        df[title+'_val_vowel_diacritic_accuracy'] = history_df['val_dense_4_accuracy']
        df[title+'_val_consonant_diacritic_accuracy'] = history_df['val_dense_5_accuracy']
    else:
        df[title+'_grapheme_root_loss'] = history_df['dense_2_loss']
        df[title+'_vowel_diacritic_loss'] = history_df['dense_3_loss']
        df[title+'_consonant_diacritic_loss'] = history_df['dense_4_loss']

        df[title+'_val_grapheme_root_loss'] = history_df['val_dense_2_loss']
        df[title+'_val_vowel_diacritic_loss'] = history_df['val_dense_3_loss']
        df[title+'_val_consonant_diacritic_loss'] = history_df['val_dense_4_loss']
        
        df[title+'_grapheme_root_accuracy'] = history_df['dense_2_accuracy']
        df[title+'_vowel_diacritic_accuracy'] = history_df['dense_3_accuracy']
        df[title+'_consonant_diacritic_accuracy'] = history_df['dense_4_accuracy']

        df[title+'_val_grapheme_root_accuracy'] = history_df['val_dense_2_accuracy']
        df[title+'_val_vowel_diacritic_accuracy'] = history_df['val_dense_3_accuracy']
        df[title+'_val_consonant_diacritic_accuracy'] = history_df['val_dense_4_accuracy']
        
        



#     df[title+'_val_loss'] = history_df['val_loss']
#     df[title+'_dice_coef'] = history_df['dice_coef']
#     df[title+'_val_dice_coef'] = history_df['val_dice_coef']

# Plot a model for each parameter
for param in actual_params:
    plot_list = [title+'_'+param for title in titles]
    df[plot_list].plot(title=param)
    plt.savefig(f'figures/{param}.jpg')
    
    


# In[3]:


df = pd.read_csv('history_dense.csv')


# In[4]:


df


# In[5]:


for i, filepath in enumerate(filepaths):
    print(filepath)
    history_df = pd.read_csv(filepath)
    if i<5:
        print(history_df[history_df['val_dense_3_accuracy']==history_df['val_dense_3_accuracy'].max()])
    else:
        print(history_df[history_df.val_dense_2_accuracy==history_df.val_dense_2_accuracy.max()])


# In[6]:


jb


# In[ ]:




