#!/usr/bin/env python3

from utils_main import *
import os

def selectmodel():

    models = {}
    for i, file in enumerate(os.listdir(models_dir)):
        if not file.endswith(".keras"): continue
        models[i] = file

    if len(models) == 0:
        print("no models found")
        exit()

    model_str = models[0]

    if len(models) > 1:
        print("Please select a model:")

        for k, f in models.items():
            print(f"\t {k}. {f}")

        try:
            selection = int(input("num: "))
        except:
            exit()

        model_str = models[selection]

    print(f"selected model: {model_str}")

    from tensorflow.keras.models import load_model
    return load_model(d_models(model_str))
