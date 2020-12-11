# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:56:22 2020

@author: Drishti
"""

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

currency_model = tf.keras.models.load_model("Currency Identification/Currency_model.h5")

print("Enter the image path:")
path = input()
if path.find("jpg") or path.find("jpeg") or path.find("png"):
    test_image = image.load_img(path, target_size = (64, 64))
    test_image.show()
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    #finding type of currency
    print("Currencies")
    proba = currency_model.predict(test_image)[0]
    #print(proba)
    l=proba.tolist()
    idxs = np.argsort(proba)[::-1][:2]
    names=[]
    prob = []
    names = ['Euro','Indian Rupees','US Dollar']
    for (label, p) in zip(names, proba):
        prob.append("{:.2f}".format(p*100))
    res = "\n".join("{} : {}%".format(x, y) for x, y in zip(names, prob))
    print(res)

    max_val = max(prob)
    max_index = prob.index(max_val)
    final_pred_curr = names[max_index]
    print("Final Predicted Currency is : ",final_pred_curr)
    print("\n")

    if final_pred_curr == 'Euro' or final_pred_curr == 'US Dollar':
        euro_usd_model = tf.keras.models.load_model("euro_usd_currency/euro_usd_denomination.h5")
        print("Denominations:")
        proba = euro_usd_model.predict(test_image)[0]
        l=proba.tolist()
        idxs = np.argsort(proba)[::-1][:2]
        names=[]
        prob = []
        names = ['1','10','100','20','200','5','50','500']
        for (label, p) in zip(names, proba):
            prob.append("{:.2f}".format(p*100))
        res = "\n".join("{} : {}%".format(x, y) for x, y in zip(names, prob))
        print(res)

    else:
        indian_model = tf.keras.models.load_model("Indian Currencies/Indian_denomination.h5")
        print("Denominations:")
        proba = indian_model.predict(test_image)[0]
        l=proba.tolist()
        idxs = np.argsort(proba)[::-1][:2]
        names=[]
        prob = []
        names = ['10','100','20','200','2000','50','500']
        for (label, p) in zip(names, proba):
            prob.append("{:.2f}".format(p*100))
        res = "\n".join("{} : {}%".format(x, y) for x, y in zip(names, prob))
        print(res)

    max_val1 = max(prob)
    max_index1 = prob.index(max_val1)
    final_pred_deno = names[max_index1]
    print("\n")
    print("The final prediciton is" )
    print(final_pred_deno , final_pred_curr)

else:
    print("Upload Image again")
    print("Run File Again")




