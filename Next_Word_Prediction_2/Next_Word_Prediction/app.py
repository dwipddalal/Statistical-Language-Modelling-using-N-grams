import gradio as gr
import gradio.inputs
import pandas as pd
import numpy as np # linear algebra
import os #interacting with input and output directories
# import tensorflow as tf #framework for creating the neural network
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from copy_of_final import LanguageModel
with open('/home/dwip.dalal/mathpro/data/model.sav','rb') as handle:
    loaded_model = pickle.load(handle)
def fn(X_test):
    sentiment = ['Do you really dislike the movie so much?','Hmm...your thoughts are neutral about the movie.','Wow! Your a big fan.']

    X_final = tuple(map(str, X_test.split(' ')))
    model = loaded_model
    result = model._best_candidate(X_final,0)
    
    return result
description = " "
here = gr.Interface(fn=fn,
                     inputs= gradio.inputs.Textbox( lines=1, placeholder=None, default="", label=None),
                     outputs='text',
                     title="Next Word Prediction",
                     description=description,
                     theme="default",
                     allow_flagging="auto",
                     flagging_dir='flagging records')
#here.launch(inline=False, share = True)
if __name__ == "__main__":
    app, local_url, share_url = here.launch(share=True)


