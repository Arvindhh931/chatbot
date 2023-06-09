import streamlit as st
from streamlit_chat import message as st_message
from preprocesser_class import Textpipeline
import datetime
from typing import Union,Optional

from spellchecker import SpellChecker
import re
import string
import pickle
import json
import nltk
from nltk.tokenize import TweetTokenizer,RegexpTokenizer,WordPunctTokenizer

import random
import pandas as pd
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity

if "history" not in st.session_state:
    st.session_state.history = []

def greeting():
    hour = datetime.datetime.now().hour
    if 0 < hour <= 11:
        return 'Hello ! good morning, How can i help you ?'
    elif 11 < hour <= 16:
        return 'Hello ! good after noon, How can i help you ?'
    elif 16 < hour <= 20:
        return 'Hello ! good evening, How can i help you ?'
    else:        
        return 'Hello ! How can i help you ?'
    
st.title(greeting())

def generate_answer():
    try:
        user_message = st.session_state.input_text
        chat_instance = Textpipeline(user_message)
        if chat_instance.get_answer() == None:
            message_bot = random.choice(chat_instance.give_apology()['unable_understand_context'])
        else:
            message_bot = chat_instance.get_answer() 
        st.session_state.history.append({"message": user_message, "is_user": True})
        st.session_state.history.append({"message": message_bot, "is_user": False})
    except Exception as e:
        pass

st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i))