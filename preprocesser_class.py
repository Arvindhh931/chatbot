from typing import Union,Optional

from spellchecker import SpellChecker
import re
import string
import gzip
import pickle
import json

import nltk
from nltk.tokenize import TweetTokenizer,RegexpTokenizer,WordPunctTokenizer

import random
import pandas as pd
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity

class Textpipeline:
    def __init__(self,rawtext:str):
        try:
            assert type(rawtext) == str,None
            self.server_down = False
            self.empty_query = False
            self.total_apologies = 0
            assert rawtext != "",None
            # lower casing, removing extra white spaces
            # original tokens
            self.__original_tokens = [word.strip() for word in rawtext.strip().lower().split(" ")]
            self.no_original_tokens = len(self.__original_tokens)
            self.__rawtext:str = " ".join(self.__original_tokens)
            self.original_text_length = len(self.__rawtext)
            # after lookup
            self.tokens_after_processing = self.processed_tokens()
            self.no_tokens_after_processing = len(self.tokens_after_processing)
            self.unique_word_count = len(self.get_unique_tokens())
            # stopwords after lookup
            self.unique_stopwords = self.all_stopwords_in_text()
            self.stopwords_count = len(self.unique_stopwords)
            self.non_stopwords_count = self.no_tokens_after_processing - self.stopwords_count
            # single word query
            self.is_text_single_token = (self.original_text_length == 1)
            # self.__only_non_english_pattern1 = '[a-z]+[^\d\W^_^a-zA-Z^\s]+[a-zA-Z]+|[^\d\W^_^a-zA-Z^\s]+'
            # special
            self.__only_special_pattern = '[\W_]+'
            self.__special_tokens = [spl_token.strip() for spl_token in re.findall('[\W]+',string=self.__rawtext) if spl_token.strip() != ""]  
            self.special_exist:bool = bool(self.__special_tokens)
            # alpha
            self.__alpha_pat = '[a-zA-Z]+'
            self.alpha_exist:bool = bool(re.findall(pattern=self.__alpha_pat,string=self.__rawtext))
            # numeric
            self.__numeric_pat = '[\d]+'
            self.numeric_exist:bool = bool(re.findall(pattern=self.__numeric_pat,string=self.__rawtext))
            self.__numeric_tokens = re.findall(pattern=self.__numeric_pat,string=self.__rawtext)
            # Non_english
            self.__only_non_english_pattern = '[^\d\W^_^a-zA-Z^\s]'
            self.non_english_tokens = self.get_non_english_words()
            # combination check
            self.alpha_numeric: bool = all([self.alpha_exist,self.numeric_exist])
            self.special_numeric: bool = all([self.special_exist,self.numeric_exist])
            self.alpha_special: bool = all([self.alpha_exist,self.special_exist])
            self.alpha_special_numeric: bool = all([self.alpha_exist,self.special_exist,self.numeric_exist])
            self.non_english:bool = bool(self.non_english_tokens)
            self.alpha_with_non_english:bool= all([self.alpha_exist,self.non_english])
        except AssertionError as A:
            self.empty_query = True
            return None
        except Exception as E:
            self.server_down = True
            return None

    def greeting(self):
        hour = datetime.datetime.now().hour
        if 0 < hour <= 11:
            return 'Hello ! good morning, How can i help you with HRMS related needs ?'
        elif 11 < hour <= 16:
            return 'Hello ! good after noon, How can i help you with HRMS related needs ?'
        elif 16 < hour <= 20:
            return 'Hello ! good evening, How can i help you with HRMS related needs ?'
        else:        
            return 'Hello ! How can i help you with HRMS related needs ?'
    
    def has_link(self,text:Optional[str]=None)->bool:
        if not text:
            text = self.__rawtext
        link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        match = re.search(pattern=link_pattern,string=text)
        return bool(match)   
    
    def has_tags(self,text:Optional[str]=None)->bool:
        if not text:
            text = self.__rawtext
        tag_pattern = r'<[^>]+>|<[^>]+/>'
        match = re.search(pattern=tag_pattern, string=text)
        return bool(match)
        
    def is_query(self,text:Optional[str]=None)->bool:
        if not text:
            text = self.__rawtext        
        return not bool(re.fullmatch(pattern='[^a-zA-Z]+[\d\W\s_]+',string=text))
            
    def is_english(self,text:Optional[str]=None)->bool:
        if not text:
            text =self.__rawtext        
        return not bool(len(re.findall(pattern='[^a-zA-Z\d\W\s_]+',string=text)))
    
    def get_non_english_words(self):
        charecters = re.findall(pattern=self.__only_non_english_pattern,string=self.__rawtext)
        non_english_words = [word for char in charecters for word in self.__original_tokens if char in word]
        return non_english_words    

    def get_all_emojis(self,text:Optional[str]=None)->Optional[list]:
        if not text:
            text = self.__rawtext
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese characters
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+",flags=re.UNICODE)
        all_emojis = emoji_pattern.findall(text)
        if len(all_emojis) > 0:
            return all_emojis
        else:
            return None
    
    def get_demojised_list(self)->Optional[list[str]]:
        if self.get_all_emojis():
            return [emoji.demojize(i).strip(":") for i in self.get_all_emojis()]
        else:
            return None

    def __get_all_slangs(self,starts_with:Optional[str]=None)->dict[str]:
        if not starts_with:
            starts_with = ''
        FILE_NAME = "./data/slang_copy.json"
        with open(FILE_NAME,'r') as file:
            all_slangs = json.load(file)
        return dict(filter(lambda x: x[0].startswith(starts_with),all_slangs.items()))
    
    def __get_all_contractions(self,starts_with:Optional[str]=None)->dict[str]:  
        if not starts_with:
            starts_with = ''
        FILE_NAME = "./data/contractions_copy.json" 
        with open(FILE_NAME,'r') as f:
            contractions = json.load(f)
        return dict(filter(lambda x: x[0].startswith(starts_with),contractions.items()))
    
    def __get_all_stopwords(self,starts_with:Optional[str]=None)->list[str]:
        if not starts_with:
            starts_with = ''
        FILE_NAME = "./data/stopwords_copy.pickle"
        with open(FILE_NAME,'rb') as file:
            stopwords = pickle.load(file)
        return list(filter(lambda x: x.startswith(starts_with),stopwords))
    
    def __get_current_directory(self):
        return os.getcwd()
    
    def get_all_tokens(self,rawtext:Optional[str]=None):
        if not rawtext:
            rawtext = self.__rawtext
        tokenizer = RegexpTokenizer(pattern='[a-zA-Z\']+[^\s\d_\W]+|[a-zA-Z\']{1}')
        regex_tokenized = tokenizer.tokenize(rawtext)

        pat1 = "([a-zA-Z]+)\s(\')\s([a-zA-Z]+)"
        pat2 = "([a-zA-Z]+)\s(\')\s([a-zA-Z]+)\s(\')\s([a-zA-Z]+)"
        repl1 = r"\1\2\3"
        repl2 = r"\1\2\3\4\5"        
        cleaned_text = " ".join(regex_tokenized) 
        
        if cleaned_text.count("'") <= 2:
            final_cleaned_text = re.sub(pattern=pat1,repl=repl1,string=cleaned_text)
            return final_cleaned_text.split()
        else:
            final_cleaned_text = re.sub(pattern=pat2,repl=repl2,string=cleaned_text)
            return final_cleaned_text.split()
    
    def get_unique_tokens(self,all_tokens:Optional[list[str]]=None)->list[str]:
        # unique tokens maintains the original order
        if not all_tokens:
            all_tokens = self.get_all_tokens()
        return all_tokens[0:1]+[token for index,token in enumerate(all_tokens) 
                                if token not in all_tokens[0:1]+all_tokens[index-1:0:-1][::-1]]    
        
    def get_analytical_features(self):
        pass
    
    def slang_lookup(self,rawtokens=None)->list[str]:
        if not rawtokens:
            rawtokens = self.get_all_tokens()
        slang = self.__get_all_slangs()
        if sum([1 if token in slang else 0 for token in rawtokens]) > 0:
            corrected_tokens = [slang[token] if token in slang else 
                               token for token in rawtokens]
            for word in corrected_tokens:
                if " " in word:
                    index = corrected_tokens.index(word)
                    corrected_tokens.pop(index)
                    for position,sub_word in enumerate(word.split()):
                        corrected_tokens.insert(index+position,sub_word)
            return corrected_tokens         
        else:
            return rawtokens 
            
    def contraction_lookup(self,all_tokens=None)->list[str]:
        if not all_tokens:
            all_tokens = self.get_all_tokens()
        contractions = self.__get_all_contractions()
        if any([True if word in contractions else False for word in all_tokens]):
            processed_tokens = [contractions[word] if word in contractions else 
                               word for word in all_tokens]
            for word in processed_tokens:
                if " " in word:
                    index = processed_tokens.index(word)
                    processed_tokens.pop(index)
                    for position,sub_word in enumerate(word.split()):
                        processed_tokens.insert(index+position,sub_word)
                else:
                    continue
            return processed_tokens
        else:
            lookup_table = {"s":"is","d" :"would","ll" : "will","ve" : "have",
                            "nt":"not","t": "not","re":'are',"m":"am"}
            all_tokens_copy = all_tokens.copy()
            for token_index,token in enumerate(all_tokens_copy):
                if bool(re.match(pattern="[a-zA-Z]+\'[a-zA-Z]+(\'[a-zA-Z]+)?",string=token)):
                    changed_tokens = [lookup_table[t.strip()] if t.strip() in lookup_table else t.strip() 
                                       for t in token.split("'")]
                    all_tokens_copy.pop(token_index)
                    for subtoken_index,sub_token in enumerate(changed_tokens):
                        all_tokens_copy.insert(token_index+subtoken_index,sub_token)
                else:
                    all_tokens_copy[token_index] = re.sub(pattern="([a-zA-Z]+)\'([a-zA-Z]+)(\'[a-zA-Z]+)",
                                                     repl=r"\1",string=token)
            final_lookedup_tokens = self.slang_lookup(rawtokens=all_tokens_copy)
            return final_lookedup_tokens

    def token_spell_check(self,incorrect_tokens:Optional[list[str]]=None,distance=3)->list[str]:
        if not incorrect_tokens:
            incorrect_tokens = self.get_all_tokens()
        spell = SpellChecker(language='en',distance=distance,case_sensitive=False)
        return [spell.correction(token) for token in incorrect_tokens if spell.correction(token) is not None]
    
    def all_stopwords_in_text(self,rawtext:Optional[str]=None)->list[str]:
        stopwords = self.__get_all_stopwords()
        if not rawtext:
            rawtext = self.__rawtext
        rawtokens = self.get_all_tokens(rawtext)
        tokens = self.slang_lookup(rawtokens=rawtokens)
        stopwords_in_text = [word for word in tokens if word in stopwords]
        return sorted(set(stopwords_in_text)) 
    
    def remove_stopwords(self,full_tokens:Optional[list[str]]=None):
        stop_words_in_text = self.stopwords_unique
        if not full_tokens:
            full_tokens = self.get_all_tokens()
        return [word for word in full_tokens if word not in stop_words_in_text]
                          
    def processed_tokens(self,text:Optional[str]=None)->list[str]:
        if not text:
            full_raw_tokens = self.get_all_tokens()
        else:
            full_raw_tokens = self.get_all_tokens(rawtext=text)
        full_raw_tokens_slang = self.slang_lookup(rawtokens=full_raw_tokens)
        expanded_tokens = self.contraction_lookup(all_tokens=full_raw_tokens_slang)
        return expanded_tokens

    def processed_text(self,remove_stopwords:bool=False)->str:
        tokens_processed = self.processed_tokens()
        if remove_stopwords:
            return " ".join(self.remove_stopwords(full_tokens=tokens_processed))
        else:
            return " ".join(tokens_processed)
        
    def load_vectorizer(self):
        FILE_NAME = './data/tfidf_vectorizer_copy.pickle'
        with open(FILE_NAME,'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return tfidf_vectorizer    
        
    def load_vectorized_arrays(self):
        # uncompress the tfidf-array
        FILE_NAME = './data/tfidf_array_copy.pkl.gz'
        with gzip.open(FILE_NAME,'rb') as f:
            compressed_array = f.read()
        # you have decompressed content while reading, now unpickle the tfidf array
        tfidf_array = pickle.loads(compressed_array)
        return tfidf_array
    
    def load_dataframe(self):
        FILE_NAME = './data/dataframe_copy.pickle'
        with open(FILE_NAME,'rb') as dataframe:
            df = pickle.load(dataframe)
        return df
        
    def save_arrays(self,arrays)->bool:
        FILE_NAME = './data/tfidf_array_copy.pkl.gz'
        compressed_array = gzip.compress(pickle.dumps(arrays))
        with open(FILE_NAME, 'wb') as file:
            file.write(compressed_tfidf_array)
    
    def give_apology(self):
        FILE_NAME = './data/apology_copy.json'
        with open(FILE_NAME, 'r') as file:
            apology = json.load(file)
        return apology
    
    def give_end_greeting(self):
        pass
    
    def get_answer(self,tfidf_vectorizer=None,tfidf_array:np.ndarray=None)->int:
        self.no_of_apologies = 0
        try:
            text = self.processed_text(remove_stopwords=False)
            if not tfidf_vectorizer:
                tfidf_vectorizer = self.load_vectorizer()
            if not tfidf_array:
                tfidf_array = self.load_vectorized_arrays()
            question_vec = tfidf_vectorizer.transform([text]).toarray()
            similarity_array = cosine_similarity(question_vec,tfidf_array).flatten()
            similarity_array[0] = 0.0
            assert similarity_array.max() > 0.30,'no_response'
            multiple_matches = False
            indices = np.where(similarity_array == similarity_array.max())[0]
            print(indices)
            if len(indices)>1:
                multiple_matches = True
            highest_score_index = int(indices[0])
            df = self.load_dataframe()
            return df['response'].iloc[highest_score_index]
        except AssertionError as A:
            return None
            
# --------------------------------------------------------------------------------------

class Conversation:
    def __init__(self):
        pass
    
    def greeting(self):
        hour = datetime.datetime.now().hour
        if 0 < hour <= 11:
            return 'Hello ! good morning, How can i help you with HRMS related needs ?'
        elif 11 < hour <= 16:
            return 'Hello ! good after noon, How can i help you with HRMS related needs ?'
        elif 16 < hour <= 20:
            return 'Hello ! good evening, How can i help you with HRMS related needs ?'
        else:        
            return 'Hello ! How can i help you with HRMS related needs ?' 
    
    def start_chatting(self):
        flag = True
        response = {'edgecase1' : "I apologize, but I didn't quite understand your question. Could you please try rephrasing it or providing me with more details?  Thanks",
               'edgecase2' : "Apologies ! I'm having trouble understanding your message in language other than English. Could you please try asking your question in English so that I can assist you better? Thanks"}
        first_message = True
        while flag:
            if first_message:
                first_message = False
                total_apologies = 0
                total_valid_queries = 0
                return self.greeting()
            raw_query = input("user : ")
            raw_query = raw_query.strip()
            # initialize_pipeline_instance
            chat_instance = textpipeline(raw_query)
            if chat_instance.server_down:
                server_down_apology = chat_instance.give_apology()['server_down']
                print(random.choice(server_down_apology))
                flag = False
            elif chat_instance.empty_query:                
                flag2 = True
                while flag2:
                    end_conv = input("conversational AI : Do you want to end conversation ?\nplease press 'Y'- yes and 'N'- No \n")
                    if end_conv.lower().startswith('y'):
                        print("conversational AI :","\ncool ! happy to have conversation with you, Take care")
                        flag,flag2 = False,False     #comes out of both while loop
                    elif end_conv.lower().startswith('n'):
                        print("conversational AI : ok great ! feel free to ask HRMS related needs")
                        flag2 = False
                    else:
                        print("conversational AI : ","I am not understanding your response\nplease follow the instructions")
            elif chat_instance.alpha_special_numeric or chat_instance.special_numeric:
                print(response['edgecase1'])
            elif chat_instance.non_english and chat.alpha_with_non_english:
                print(response['edgecase2'])                
            else:
                total_valid_queries += 1
                print(total_valid_queries)
                if chat_instance.get_answer() == None:
                    if total_apologies == 0:
                        context_apology = chat_instance.give_apology()['unable_understand_context']
                        print("conversational AI : ",random.choice(context_apology))
                        total_apologies += 1
                        print(total_apologies)
                    elif total_apologies >= 1:
                        contact_representative = chat_instance.give_apology()['contact_human']
                        print("conversational AI : ",random.choice(contact_representative))
                        total_apologies += 1
                        print(total_apologies)
                    else:
                        server_down_apology = chat_instance.give_apology()['server_down']
                        print("conversational AI : ",random.choice(server_down_apology))
                else:
                    print("conversational AI : ",chat_instance.get_answer())
                    
# --------------------------------------------------------------------------------------------------