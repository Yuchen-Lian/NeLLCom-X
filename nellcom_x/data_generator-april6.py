#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import itertools
import numpy as np
import pandas as pd
import random
random.seed(10)


# In[2]:


noun_new = np.array(['Name_'+str(i) for i in range(1, 11)])
verb_new = ['Verb_'+str(i) for i in range(1, 9)]

# 18 novel content words (8 verbs and 10 nouns) 


# In[3]:

def create_meaning_list(noun_new, verb_new, order='Aap'):
    permu = list(itertools.permutations(range(len(noun_new)), 2))
    meaning_new = []
    for v in verb_new:
        for p in permu:
            if order == 'Aap':
                m = v + ' ' + noun_new[p[0]] + ' ' + noun_new[p[-1]]
            elif order == 'Apa':
                m = v + ' ' + noun_new[p[-1]] + ' ' + noun_new[p[0]]
            elif order == 'apA':
                m = noun_new[p[0]] + ' ' + noun_new[p[-1]] + ' ' + v
            elif order == 'aAp':
                # tacl submitted version
                m = noun_new[p[0]] + ' ' + v + ' ' + noun_new[p[-1]]
            meaning_new.append(m)
    return meaning_new


# In[9]:


class new_Language2():        
    def __init__(self, random_seed, meaning, language_name, use_marker=True, marker_percent=1, free_order=False, free_percent=0.5, meaning_order = 'aAp'):
        if use_marker:
            self.marker_percent = marker_percent
        else:
            self.marker_percent = 0

        self.random_seed = random_seed
            
        self.lang_samples = None
        self.language_name = language_name
        self.trajectories = None
        self.meaning_order = meaning_order
        
        if free_order:
            self.free_percent = free_percent
        else:
            self.free_percent = 0
        
        self.marker = 'mk'
        self.generate_language(meaning)

    def generate_language(self, meaning):
        self.trajectories = meaning
        lang_samples = []
        new_meaning = []
        random.seed(self.random_seed)

        if self.meaning_order == 'Aap':
            # tacl v1 rerun version
            # m = v + ' ' + noun_new[p[0]] + ' ' + noun_new[p[-1]]
            for m in meaning:
                v, sub, obj = m.split()
                uttr_fix = sub + ' ' + obj + ' ' + v
                uttr_fix_mk = sub + ' ' + obj + ' ' + self.marker + ' ' + v
                uttr_reverse = obj + ' ' + sub + ' ' + v
                uttr_reverse_mk = obj + ' ' + self.marker + ' ' + sub + ' ' + v
                # new_m = sub + '_sub' + ' ' + v + ' ' + obj + '_obj'
                new_m = m
                new_meaning.append(new_m.upper())

                rand_free = random.random()
                rand_mark = random.random()
                if rand_free <= self.free_percent and rand_mark <= self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_reverse_mk])
                elif rand_free <= self.free_percent and rand_mark > self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_reverse])
                elif rand_free > self.free_percent and rand_mark <= self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_fix_mk])
                elif rand_free > self.free_percent and rand_mark > self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_fix])
                else:
                    print('error')

        elif self.meaning_order == 'Apa':
            # m = v + ' ' + noun_new[p[-1]] + ' ' + noun_new[p[0]]
            for m in meaning:
                v, obj, sub = m.split()
                uttr_fix = sub + ' ' + obj + ' ' + v
                uttr_fix_mk = sub + ' ' + obj + ' ' + self.marker + ' ' + v
                uttr_reverse = obj + ' ' + sub + ' ' + v
                uttr_reverse_mk = obj + ' ' + self.marker + ' ' + sub + ' ' + v
                # new_m = sub + '_sub' + ' ' + v + ' ' + obj + '_obj'
                new_m = m
                new_meaning.append(new_m.upper())

                rand_free = random.random()
                rand_mark = random.random()
                if rand_free <= self.free_percent and rand_mark <= self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_reverse_mk])
                elif rand_free <= self.free_percent and rand_mark > self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_reverse])
                elif rand_free > self.free_percent and rand_mark <= self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_fix_mk])
                elif rand_free > self.free_percent and rand_mark > self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_fix])
                else:
                    print('error')

        elif self.meaning_order == 'apA':
            # m = noun_new[p[0]] + ' ' + noun_new[p[-1]] + ' ' + v
            for m in meaning:
                sub, obj, v = m.split()
                uttr_fix = sub + ' ' + obj + ' ' + v
                uttr_fix_mk = sub + ' ' + obj + ' ' + self.marker + ' ' + v
                uttr_reverse = obj + ' ' + sub + ' ' + v
                uttr_reverse_mk = obj + ' ' + self.marker + ' ' + sub + ' ' + v
                # new_m = sub + '_sub' + ' ' + v + ' ' + obj + '_obj'
                new_m = m
                new_meaning.append(new_m.upper())

                rand_free = random.random()
                rand_mark = random.random()
                if rand_free <= self.free_percent and rand_mark <= self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_reverse_mk])
                elif rand_free <= self.free_percent and rand_mark > self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_reverse])
                elif rand_free > self.free_percent and rand_mark <= self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_fix_mk])
                elif rand_free > self.free_percent and rand_mark > self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_fix])
                else:
                    print('error')

        elif self.meaning_order == 'aAp':
            # tacl submitted version
            # m = noun_new[p[0]] + ' ' + v + ' ' + noun_new[p[-1]]
            for m in meaning:
                sub, v, obj = m.split()
                uttr_fix = sub + ' ' + obj + ' ' + v
                uttr_fix_mk = sub + ' ' + obj + ' ' + self.marker + ' ' + v
                uttr_reverse = obj + ' ' + sub + ' ' + v
                uttr_reverse_mk = obj + ' ' + self.marker + ' ' + sub + ' ' + v
                # new_m = sub + '_sub' + ' ' + v + ' ' + obj + '_obj'
                new_m = m
                new_meaning.append(new_m.upper())

                rand_free = random.random()
                rand_mark = random.random()
                if rand_free <= self.free_percent and rand_mark <= self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_reverse_mk])
                elif rand_free <= self.free_percent and rand_mark > self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_reverse])
                elif rand_free > self.free_percent and rand_mark <= self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_fix_mk])
                elif rand_free > self.free_percent and rand_mark > self.marker_percent:
                    lang_samples.append([new_m.upper(), uttr_fix])
                else:
                    print('error')

        # for m in meaning:
        #     sub, v, obj = m.split()
        #     uttr_fix = sub + ' ' + obj + ' ' + v
        #     uttr_fix_mk = sub + ' ' + obj + ' ' + self.marker + ' ' + v
        #     uttr_reverse = obj + ' ' + sub + ' ' + v
        #     uttr_reverse_mk = obj + ' ' + self.marker + ' ' + sub + ' ' + v
        #     # new_m = sub + '_sub' + ' ' + v + ' ' + obj + '_obj'
        #     new_m = m
        #     new_meaning.append(new_m.upper())
        #
        #     rand_free = random.random()
        #     rand_mark = random.random()
        #     if rand_free <= self.free_percent and rand_mark <= self.marker_percent:
        #         lang_samples.append([new_m.upper(), uttr_reverse_mk])
        #     elif rand_free <= self.free_percent and rand_mark > self.marker_percent:
        #         lang_samples.append([new_m.upper(), uttr_reverse])
        #     elif rand_free > self.free_percent and rand_mark <= self.marker_percent:
        #         lang_samples.append([new_m.upper(), uttr_fix_mk])
        #     elif rand_free > self.free_percent and rand_mark > self.marker_percent:
        #         lang_samples.append([new_m.upper(), uttr_fix])
        #     else:
        #         print('error')
                    
        self.lang_samples = lang_samples
        self.trajectories = new_meaning
        return


# In[14]:


class new_Dataset():
    def __init__(self, Language):
        
        self.language_samples = Language.lang_samples
        self.trajectories = Language.trajectories
        # self.free_order = Language.free_order
        
        self.path_prefix = f'./data_expand/{Language.language_name}'
        if not os.path.exists(self.path_prefix):
            os.mkdir(self.path_prefix)
        
    def write_to_file(self, fname='meaning_phrase.txt'):
        
        np_lang = np.array(self.language_samples)
        df_lang = pd.DataFrame(np_lang)
        self.df_lang = df_lang
        
        if not os.path.exists(self.path_prefix ):
            os.mkdir(self.path_prefix )

        df_lang.to_csv(f'{self.path_prefix}/{fname}', sep='\t', header=0, index=0)
        return


# Language_free_op_default = new_Language2(meaning_new, language_name='free_op',
#                                            use_marker=True, marker_percent=0.667, free_order=True, free_percent=0.5)
# ds_Language_free_op_default = new_Dataset(Language_free_op_default).write_to_file('meaning_phrase_0.txt')
#
# Language_free_op_1 = new_Language2(meaning_new, language_name='free_op',
#                                            use_marker=True, marker_percent=0.2, free_order=True, free_percent=0.5)
# ds_Language_free_op_1 = new_Dataset(Language_free_op_1).write_to_file('meaning_phrase_1.txt')

meaning_new = create_meaning_list(noun_new, verb_new, order='aAp')
p_mks = [0.5, 0.2, 0.8]
p_osvs = [0.5, 0.2, 0.8]

p_mks = [0.95, 0.5, 0.05]
p_osvs = [0.95, 0.5, 0.05]


p_mks = [0.67]
p_osvs = [0.25, 0.75]

p_mks = [0.75]
p_osvs = [0.5]

p_mks = [0.67]
p_osvs = [0.50]


p_mks = [0.2]
p_osvs = [0.8]

p_mks = [0.8]
p_osvs = [0.5]
randseed = 4
suffix = f'_v{randseed}'
combo = itertools.product(p_mks, p_osvs)
for p_mk, p_osv in combo:
    Language_free_op = new_Language2(randseed, meaning_new, language_name='free_op',
                                       use_marker=True, marker_percent=p_mk, free_order=True, free_percent=p_osv, meaning_order='aAp')
    ds_Language_free_op = new_Dataset(Language_free_op).write_to_file(f'meaning_phrase_mk{int(p_mk*100)}%_osv{int(p_osv*100)}%{suffix}.txt')

#p_mks = [0.67]
#randseed = 4
#suffix = f'_v{randseed}' if randseed != 0 else ''
#for p_mk in p_mks:
#    Language_free_op = new_Language2(randseed, meaning_new, language_name='fix_op',
#                                       use_marker=True, marker_percent=p_mk, free_order=False, free_percent=0, meaning_order='aAp')
#    ds_Language_free_op = new_Dataset(Language_free_op).write_to_file(f'meaning_phrase_mk{int(p_mk*100)}%{suffix}.txt')


# In[16]:


# Language_head_first_fix2 = new_Language2(meaning_new, language_name='fix2',
#                                        use_marker=False, free_order=False)
#
# Language_head_first_fix_mk2 = new_Language2(meaning_new, language_name='fix_mk2',
#                                            use_marker=True, marker_percent=1, free_order=False)
#
# Language_head_first_free_mk2 = new_Language2(meaning_new, language_name='free_mk2',
#                                            use_marker=True, marker_percent=1, free_order=True, free_percent=0.5)
#
#
# ds_Language_head_first_fix2 = new_Dataset(Language_head_first_fix2).write_to_file()
# ds_Language_head_first_fix_mk2 = new_Dataset(Language_head_first_fix_mk2).write_to_file()
# ds_Language_head_first_free_mk2 = new_Dataset(Language_head_first_free_mk2).write_to_file()


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


# class new_Language():        
#     def __init__(self, meaning, language_name, use_marker=1, free_order=False):
#         self.use_marker = use_marker == 1
#         self.lang_samples = None
#         self.language_name = language_name
#         self.trajectories = None
#         self.free_order = free_order
#         self.marker = 'mk'
#         self.generate_language(meaning)
        
    
#     def generate_language(self, meaning):
#         self.trajectories = meaning
#         lang_samples = []
#         new_meaning = []
        
#         for m in meaning:
#             sub, v, obj = m.split()
#             uttr_fix = sub + ' ' + obj + ' ' + v
#             uttr_fix_mk = sub + ' ' + obj + ' ' + self.marker + ' ' + v
#             uttr_reverse_mk = obj + ' ' + self.marker + ' ' + sub + ' ' + v
#             # new_m = sub + '_sub' + ' ' + v + ' ' + obj + '_obj'
#             new_m = m
#             new_meaning.append(new_m.upper())
            
#             if self.free_order:
#                 if random.random() > 0.5:
#                     lang_samples.append([new_m.upper(), uttr_reverse_mk])
#                 else:
#                     lang_samples.append([new_m.upper(), uttr_fix_mk])
#             else:
#                 if self.use_marker:
#                     lang_samples.append([new_m.upper(), uttr_fix_mk])
#                 else:
#                     lang_samples.append([new_m.upper(), uttr_fix])
                    
#         self.lang_samples = lang_samples
#         self.trajectories = new_meaning
#         return


# In[9]:


# Language_head_first_fix_mk = new_Language(meaning_new, language_name='fix_mk')
# Language_head_first_fix = new_Language(meaning_new, language_name='fix', use_marker=0)
# Language_head_first_free_mk = new_Language(meaning_new, language_name='free_mk', free_order=True)


# In[11]:


# ds_Language_head_first_fix_mk = new_Dataset(Language_head_first_fix_mk).write_to_file()
# ds_Language_head_first_fix = new_Dataset(Language_head_first_fix).write_to_file()
# ds_Language_head_first_free_mk = new_Dataset(Language_head_first_free_mk).write_to_file()


# In[2]:




