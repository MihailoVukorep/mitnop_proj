#!/usr/bin/env python3

from utils_main import *
import os

def selectmodel():
    i = 0
    models = {}
    for file in os.listdir(models_dir):
        if not file.endswith(".keras"):
            continue
        i += 1
        models[i] = file
    print(models)

    if len(models) == 0:
        print("no models found")
        exit()

    model_str = models[1]

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
