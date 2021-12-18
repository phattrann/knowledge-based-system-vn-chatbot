# -*- coding: utf-8 -*-

import os
import errno
import shutil
import time
import random
import re
import sys
import traceback
import string
import json

import joblib
import pandas as pd
import numpy as np

from random import choice
from preprocessing import *
from utilities import *

class codebot:
    def __init__(self):
        self.__path = os.path.dirname(os.path.abspath(__file__))
        self.__load_KB()
        self.__load_models()

        self.raw_stack = []
        self.vec_stack = []
        
        self.intent = None          # Intent of user
        self.context = 'general'         # Context is for looking up the exact language in kb

        self.low_conf_accept = False
    
    def __repr__(self):
        return '<%s.%s object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )
    
    def __load_KB(self):
        self.kb = read_json(f'{self.__path}/model/knowledge.json')
        self.rb = read_json(f'{self.__path}/model/response.json')
        self.ir = read_json(f'{self.__path}/model/intent_rules.json')
        self.cr = read_json(f'{self.__path}/model/context_rules.json')

    def __load_models(self):
        # Preprocessing Model:
        self.preprocessor = joblib.load(f'{self.__path}/model/preprocessing_1.bin')

        # Linear Support Vector Classifiers:
        self.intent_clf = joblib.load(f'{self.__path}/model/model_intent_1.bin')
        self.context_clf = joblib.load(f'{self.__path}/model/model_context_1.bin')
    
    def __read_input(self, input_str):
        self.raw_stack.append(input_str)
        self.vec_stack.append(self.preprocessor.transform(pd.Series(input_str)))
    
    def __clean_up(self):
        if self.intent != 'low_conf':
            self.raw_stack.pop()
            self.vec_stack.pop()

    def __get_intent(self):
        X = self.vec_stack[-1]
        return {
            'name': self.intent_clf.predict(X)[0],
            'conf': np.max(self.intent_clf.decision_function(X))
        }
    
    def __get_context(self):
        X = self.vec_stack[-1]
        return {
            'name': self.context_clf.predict(X)[0],
            'conf': np.max(self.context_clf.decision_function(X))
        }
    
    def __parse_rb(self, response):
        if '$' not in response:
            return response[0].upper() + response[1:]
        replaces = dict(
            (w, choice(self.rb[w])) 
            for w in response.translate(str.maketrans('', '', string.punctuation.replace('$',''))).split() 
            if w[0] == '$' and w in self.rb)
        for key in replaces:
            response = response.replace(key, replaces[key])
        return self.__parse_rb(response)

    def __intent_infer(self, cur_intent):
        if self.intent in self.ir and cur_intent in self.ir[self.intent]:
            return self.ir[self.intent][cur_intent]
        return cur_intent

    def __context_infer(self, cur_context):
        if self.context in self.cr and cur_context in self.cr[self.context]:
            return self.cr[self.context][cur_context]
        return cur_context
    
    def __get_sorted_tfidf_terms(self, n=None):
        feature_array = np.array(self.preprocessor._vectorizer.get_feature_names())
        tfidf = self.vec_stack[-1].toarray()[0]
        tfidf_sorting = np.argsort(tfidf).flatten()[::-1]
        if n is not None:
            return feature_array[tfidf_sorting][:n]
        return feature_array[tfidf_sorting][tfidf[tfidf_sorting] > 0.0]
    
    def __get_theory_answer(self):
        
        # Searching KB for terms:
        tokens = self.raw_stack[-1].lower().translate(str.maketrans('', '', string.punctuation.replace('+',''))).split()
        theory_terms = []
        for i in range(len(tokens)):
            for c in range(1, 4):
                if i+c == len(tokens)+1: 
                    break
                tmp = " ".join([tokens[i+j] for j in range(c)])
                if tmp in self.kb and (self.context == 'general' or self.context in self.kb[tmp]):
                    theory_terms.append(tmp)

        # Replace $obj in response:
        rb_intent = self.rb[self.intent]
        response = choice(rb_intent[self.context])
        knowledge = []

        # print("tokens",tokens)

        if len(theory_terms) < response.count('$obj'):
            self.intent = 'other'
            response = choice(rb_intent['null'])
        elif len(theory_terms) > response.count('$obj'):
            self.intent = 'other'
            return self.__get_irrelevant_answer()
        else:
            tmp = ""
            for term in theory_terms:
                tmp = term
                response = response.replace('$obj', term, 1)
                
                if self.context == 'general':
                    if 'general' in self.kb[term] and self.intent in self.kb[term][self.context]:
                        knowledge.extend(self.kb[term][self.context][self.intent])
                        
                        if len(self.kb[term][self.context]['__source__']) > 0:
                            knowledge.append("\n\nNguồn: "+'\n'.join(self.kb[term][self.context]['__source__']) + "\n\n")
                    else:
                        for context in self.kb[term]:
                            if self.intent in self.kb[term][context]:
                                knowledge.extend(self.kb[term][context][self.intent])
                            else:
                                knowledge = []
                                break

                elif self.context != 'general' and 'general' in self.kb[term] and self.intent in self.kb[term]['general']:
                    knowledge.extend(self.kb[term]['general'][self.intent])

                    if self.intent in self.kb[term][self.context]:
                        knowledge.extend(self.kb[term][self.context][self.intent])
                        
                        if "__source__" in self.kb[term][self.context] and len(self.kb[term][self.context]['__source__']) > 0:
                            knowledge.append("\n\nNguồn: "+'\n'.join(self.kb[term][self.context]['__source__'])+ "\n\n")
                    else:
                        knowledge = []
                        break

            if len(knowledge) == 0:
                self.intent = 'other'
                response = choice(rb_intent['null'])
                response = response.replace('$obj', tmp, 1)

        print("tokens",tokens)
        print("intent",self.intent)
        print("theory",theory_terms)
        print("context",self.context)            
        # print(knowledge)
        answer = [response]
        answer.extend(knowledge)
        
        return [self.__parse_rb(sen) for sen in answer]

    def __get_irrelevant_answer(self):
        answer = ''
        rb_intent = self.rb[self.intent]
        
        if isinstance(rb_intent, list):
            answer = choice(rb_intent)
        else:
            answer = choice(rb_intent[self.context])

        if isinstance(answer, list):
            return [self.__parse_rb(sen) for sen in answer]
        return [self.__parse_rb(answer)]


    def __get_low_conf_answer(self, original_intent):
        answer = ''
        rb_intent_lc = self.rb['low_conf'][original_intent]
        
        if isinstance(rb_intent_lc, list):
            answer = choice(rb_intent_lc)
        else:
            answer = rb_intent_lc

        if isinstance(answer, list):
            return [self.__parse_rb(sen) for sen in answer]
        return [self.__parse_rb(answer)]

    def __get_final_answer(self):
        # Intent:
        intent = self.__get_intent()
        if intent['conf'] >= 0.0 or self.low_conf_accept:
            self.low_conf_accept = False
            self.intent = self.__intent_infer(intent['name'])
        elif intent['name'] in self.rb['low_conf']:
            self.intent = self.__intent_infer('low_conf')
        else:
            self.intent = 'other'

        # Context:
        context = self.__get_context()
        if context['conf'] >= 0.0:
            self.context = self.__context_infer(context['name'])
        else:
            self.context = self.__context_infer('low_conf')

        # Low_conf:
        if self.intent == "agree":
            self.low_conf_accept = True
        else:
            self.low_conf_accept = False

        # print(self.intent, self.context)
        
        # Build answer:
        if self.intent == 'low_conf':
            final_answer = self.__get_low_conf_answer(intent['name'])
        elif self.intent in ('define', 'apply', 'compare'):
            final_answer = self.__get_theory_answer()
            # print(final_answer)
        else:
            final_answer = self.__get_irrelevant_answer()
        
        print(intent, self.intent, context, self.context, self.low_conf_accept, sep='\n')

        # Only allow 1 follow-up question, hence we need to reset context:
        if self.context != context['name']:
            self.context = context['name']
        
        self.__clean_up()
        return final_answer
        
    def send_replies(self, input_str):
        self.__read_input(input_str)
        # print(self.__get_sorted_tfidf_terms())
        answers = []
        answers.extend(self.__get_final_answer())
        if len(self.raw_stack) and self.intent == 'agree':
            answers.extend(self.__get_final_answer())

        if self.intent == 'other':
            self.context = 'general'

        a = " ".join(answers)
        # return json.dumps(, ensure_ascii=False)
        return a

from flask import Flask, render_template, request

if __name__ == "__main__":
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

    bot_instance = codebot() 
    
    app = Flask(__name__, template_folder='../../template')
    app.static_folder = '../../static'
    @app.route("/")
    def home():
        return render_template("index.html")
    @app.route("/get")
    def get_bot_response():
        userText = request.args.get('msg')
        # print("final: ", bot_instance.send_replies(userText))
        return bot_instance.send_replies(userText)
    if __name__ == "__main__":
        app.run()
    
