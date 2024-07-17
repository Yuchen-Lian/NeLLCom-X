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


class new_Language_v3():        
    def __init__(self, meaning, language_name, use_marker=True, marker_percent_osv=1, marker_percent_sov=1, free_order=False, free_percent=0.5, meaning_order = 'aAp'):
        if use_marker:
            self.marker_percent_osv = marker_percent_osv
            self.marker_percent_sov = marker_percent_sov
        else:
            self.marker_percent_osv = 0
            self.marker_percent_sov = 0
            
        self.lang_samples = None
        self.language_name = language_name
        self.trajectories = None
        self.meaning_order = meaning_order
        
        # free_percent = osv_percent
        if free_order:
            self.free_percent = free_percent
        else:
            self.free_percent = 0
        
        self.marker = 'mk'
        self.generate_language_v3(meaning)

    def m_to_uttr(self, sub, v, obj, lang_samples, new_m):
        uttr_fix = sub + ' ' + obj + ' ' + v
        uttr_fix_mk = sub + ' ' + obj + ' ' + self.marker + ' ' + v
        uttr_reverse = obj + ' ' + sub + ' ' + v
        uttr_reverse_mk = obj + ' ' + self.marker + ' ' + sub + ' ' + v

        rand_free = random.random()
        rand_mark_osv = random.random()
        rand_mark_sov = random.random()
        if rand_free <= self.free_percent:
            if rand_mark_osv <= self.marker_percent_osv:
                lang_samples.append([new_m.upper(), uttr_reverse_mk])
            else:
                lang_samples.append([new_m.upper(), uttr_reverse])
        else:
            if rand_mark_sov <= self.marker_percent_sov:
                lang_samples.append([new_m.upper(), uttr_fix_mk])
            else:
                lang_samples.append([new_m.upper(), uttr_fix])

        return


    def generate_language_v3(self, meaning):
        self.trajectories = meaning
        lang_samples = []
        new_meaning = []

        if self.meaning_order == 'Aap':
            for m in meanings:
                v, sub, obj = m.split()
                self.m_to_uttr(sub, v, obj, lang_samples, m)
                new_meaning.append(m.upper())

        elif self.meaning_order == 'Apa':
            for m in meaning:
                v, obj, sub = m.split()
                self.m_to_uttr(sub, v, obj, lang_samples, m)
                new_meaning.append(m.upper())

        elif self.meaning_order == 'apA':
            for m in meanings:
                sub, obj, v = m.split()
                self.m_to_uttr(sub, v, obj, lang_samples, m)
                new_meaning.append(m.upper())
    
        elif self.meaning_order == 'aAp':
            for m in meaning:
                sub, v, obj = m.split()
                self.m_to_uttr(sub, v, obj, lang_samples, m)
                new_meaning.append(m.upper())

        self.lang_samples = lang_samples
        self.trajectories = new_meaning

        return

    def generate_language(self, meaning):
        self.trajectories = meaning
        lang_samples = []
        new_meaning = []

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
    def __init__(self, Language, path_folder='./data_expand/', path_suffix=''):
        
        self.language_samples = Language.lang_samples
        self.trajectories = Language.trajectories
        # self.free_order = Language.free_order
        
        self.path_prefix = f'{path_folder}{Language.language_name}{path_suffix}'
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


meaning_new = create_meaning_list(noun_new, verb_new, order='aAp')
p_mks_osv = [0.75]
p_mks_sov = [0.25]
p_osvs = [0.5]

combo = itertools.product(p_osvs, p_mks_osv, p_mks_sov)

for p_osv, p_mk_osv, p_mk_sov in combo:
    Language_free_op = new_Language_v3(meaning_new, language_name='free_op',
                                       use_marker=True, marker_percent_osv=p_mk_osv, marker_percent_sov=p_mk_sov, free_order=True, free_percent=p_osv, meaning_order='aAp')
    ds_Language_free_op = new_Dataset(Language_free_op, path_suffix='_v2').write_to_file(f'meaning_phrase_osv{int(p_osv*100)}%_mkOSV{int(p_mk_osv*100)}%_mkSOV{int(p_mk_sov*100)}%.txt')



