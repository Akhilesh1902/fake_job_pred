import re
import string
import math
import pandas as pd
from nltk.cluster import cosine_distance
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.corpus import wordnet
import category_encoders as ce
import pickle


def handle(e):
    print("Exception raised in " + e)

stop_words = set(stopwords.words('english'))

def missing_values(data):
    print('Handling Missing Data')
    print(data)
    # data =  data.iloc[[0]]
    # print(data)
    # print(data.to_numpy())
    try:
        # data['location'].fillna('no info', inplace = True)
        data['location'] = data['location'].fillna('no info')
        withoutcomma = data[~data['location'].str.contains(",", na=False)].index
        withcomma = data[data['location'].str.contains(",", na=False)].index
        data.loc[withcomma, 'country'] = data.loc[withcomma, 'location'].str.split(',').str[0].str.strip()
        data.loc[withoutcomma, 'country'] = data.loc[withoutcomma, 'location'].str.strip()
        """2.salary range"""

        # data['salary_range'].fillna('0-0', inplace = True)
        # data['salary_range'] = data['salary_range'].fillna('no info')

        # data.loc[data['salary_range'].str.contains(r'[a-zA-Z]', na=False), 'salary_range'] = '0-0'

        # # Extract minimum and maximum salary using regex
        # salary_split = data['salary_range'].str.extract(r'(\d+)-(\d+)')

        # # Assign extracted values
        # data['minimum_salary'] = salary_split[0].fillna(data['salary_range'])
        # data['maximum_salary'] = salary_split[1].fillna(data['salary_range'])


        """3. All other categorical columns and remaining numeric columns."""

        columns = data.columns
        for i in columns:
            if(data[i].isna().any()):
                if(data[i].dtypes == 'object'):
                    # data[i].fillna('no info', inplace = True)
                    data[i] = data[i].fillna('no info')
                    data[i] = data[i].str.lower()

                else:
                    # data[i].fillna(0, inplace = True)
                    data[i] = data[i].fillna(0)


        data.drop(['location'], axis = 1, inplace = True)
        # data.drop(['salary_range', 'location'], axis = 1, inplace = True)
        # print(data.to_numpy())
        # print('for loop')
        # for item in data.to_numpy()[0]:
        #     print('inside for loop')
        #     print(item)
        return data
    except Exception as e:
        print('exceptions  ')
        print(e)
        handle('missing data handling process')

def texthandling(data,idx):
        print('Text Handling')
        # print(data.to_dict())
        # print(data.isna().sum())
        # print(data.columns.values)
        # print(set(stopwords.words('english')))
        # return
        try:
            '''
            This function is for handling text data columns company profile,
            description, requirements, benefits are there is multiple text in
            those columns we need to do something about them.
            '''
            stop_words = set(stopwords.words('english'))
            print(data)
            print(data.shape[0])
            # for i in range(0, data.shape[0]):
            print('training : ', 0)
            print('here')
            print(data['company_profile'])
            print(data.loc[idx, 'company_profile'])
            data.loc[idx, 'company_profile'] = removeuncessary(data.loc[idx,
                                                        'company_profile'])
            print('trainin 0.1')
            
            data.loc[idx, 'description'] = removeuncessary(data.loc[idx,
                                                            'description'])
            data.loc[idx, 'requirements'] = removeuncessary(data.loc[idx,
                                                            'requirements'])
            data.loc[idx, 'benefits'] = removeuncessary(data.loc[idx,
                                                                'benefits'])
            data.loc[idx, 'title'] = removeuncessary(data.loc[idx, 'title'])
            data.loc[idx, 'department'] = removeuncessary(data.loc[idx,
                                                                'department'])
            data.loc[idx, 'industry'] = removeuncessary(data.loc[idx,
                                                                'industry'])
            data.loc[idx, 'function'] = removeuncessary(data.loc[idx,
                                                                'function'])
            print('trainin 0.1')
            words = str(data.loc[idx, 'company_profile'])
            if(words == 'no info'):
                data.loc[idx, 'company_profile_word_count'] = 0
            else:
                data.loc[idx, 'company_profile_word_count'] = len(
                                                                words.split())

            words = str(data.loc[idx, 'benefits'])
            if(words == 'no info'):
                data.loc[idx, 'benefits_word_count'] = 0
            else:
                data.loc[idx, 'benefits_word_count'] = len(words.split())

            data.loc[idx, 'title_and_job_similarity'] = synonym_relation(
                            data.loc[idx, 'title'], data.loc[idx,
                                                            'description'])

            data.loc[idx, 'title_and_req_similarity'] = synonym_relation(
                            data.loc[idx, 'title'], data.loc[idx,
                                                            'requirements'])

            data.loc[idx, 'profile_and_job_similarity'] = synonym_relation(
                    data.loc[idx, 'company_profile'], data.loc[idx,
                                                            'description'])

            data.loc[idx, 'profiel_and_req_similarity'] = synonym_relation(
                    data.loc[idx, 'company_profile'], data.loc[idx,
                                                            'requirements'])

            data.loc[idx,
            'title_and_department_syn_similarity'] = synonym_relation(
                            data.loc[idx, 'title'], data.loc[idx, 'department'])

            data.loc[idx,
            'title_and_industry_syn_similarity'] = synonym_relation(
                                data.loc[idx, 'title'],data.loc[idx, 'industry'])

            data.loc[idx,
            'title_and_function_syn_similarity'] = synonym_relation(
                                data.loc[idx, 'title'], data.loc[idx, 'function'])

            data.loc[idx,
            'industry_and_department_syn_similarity'] = synonym_relation(
                        data.loc[idx, 'industry'], data.loc[idx, 'department'])

            data.loc[idx,
            'function_and_department_syn_similarity'] = synonym_relation(
                        data.loc[idx, 'function'], data.loc[idx, 'department'])
            data.loc[idx,
            'industry_and_function_syn_similarity'] =synonym_relation(
                            data.loc[idx, 'industry'], data.loc[idx, 'function'])

            for i in ['title_and_job_similarity', 'title_and_req_similarity',
                    'profile_and_job_similarity', 'profiel_and_req_similarity',
                    'title_and_department_syn_similarity',
                    'title_and_industry_syn_similarity',
                    'title_and_function_syn_similarity',
                    'function_and_department_syn_similarity',
                    'industry_and_department_syn_similarity',
                    'industry_and_function_syn_similarity']:

                    # data[i].fillna(0, inplace = True)
                    data[i] = data[i].fillna(0)


            data.drop(['company_profile', 'benefits', 'description',
                       'requirements', 'title', 'department', 'industry',
                       'function',], axis = 1, inplace = True)
            return data
        except Exception as e:
            print(e)
            handle('Text handling process')

def stopwordsremove(text):
    stop_words = set(stopwords.words('english'))
    try:
        word_token = word_tokenize(text)
        ps = PorterStemmer()
        filtered = [ps.stem(w.lower())
                    for w in word_token if not w in stop_words]
        return filtered
    except Exception as e:
        # print('error in stopword removal')
        # print(e)
        handle('stop words removing')


def removeuncessary(text):
    # print('removing unnecessory text : ', text)
    try:
        '''
        1. removing punctuations,
        2. removing numbered words,
        3. removing unknown characters

        '''
        text = re.sub('[%s]'%re.escape(string.punctuation), '', str(text))
        text = re.sub('\w*\d\w*', '', str(text))
        text = re.sub('[^a-zA-Z ]+', ' ', str(text))

        return text
    except Exception as e:
        handle('removing unnecessary text')


def synonym_relation(text1, text2):
    try:
        if(text1 == 'no info' or text2 == 'no info'):
            return 0
        else:
            text1 = stopwordsremove(text1)
            text2 = stopwordsremove(text2)
            syn_set = set()
            count  = 0
            if(len(text1) == 0 or len(text2) == 0):
                return 0
            if(len(text1) < len(text2)):
                for word in text2:
                    for syn in wordnet.synsets(word):
                        for l in syn.lemmas():
                            syn_set.add(l.name())

                for word in text1:
                    if word in syn_set:
                            count += 1
                return (count / len(text1))
            else:
                for word in text1:
                    for syn in wordnet.synsets(word):
                        for l in syn.lemmas():
                            syn_set.add(l.name())

                for word in text2:
                    if word in syn_set:
                            count += 1
                return (count / len(text2))
    except Exception as e:
        # print(e)
        handle('synonym relation finding process')


def categorical_cols_train(data):
    try:
        print('Categorical Encoding')
        encoder = ce.BinaryEncoder(cols = ['employment_type',
          'required_experience', 'required_education', 'country'])
        newdata = encoder.fit_transform(data)
        pickle.dump( encoder, open( "model/encoder.p", "wb" ) )
        return newdata
    except Exception as e:
        handle('categorical column handling')

def categorical_cols_test(data):
    print('Categorical Encoding')
    try:
        encoder = pickle.load( open( "model/encoder.p", "rb" ) )
        newdata = encoder.transform(data)
        return newdata
    except Exception as e:
        handle('categorical columns handling for testing process')
