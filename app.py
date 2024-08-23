### Imports for Modules ### 
import gradio as gr
import os
import torch
from typing import Tuple, Dict
from timeit import default_timer as timer

### Functional Imports
from predictor import predictionMaker

exampleList = [["examples/" + example] for example in os.listdir("examples")]

title = "Kidney CT Analyzer"
description = "Trained a ViT to classify images of Based on [CT KIDNEY DATASET](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/data)."


# Create the Gradio demo
demo = gr.Interface(fn=predictionMaker, 
                    inputs=[gr.Dropdown(["ViT","EfficientNet"]),gr.Image(type="pil")], 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=exampleList, 
                    title=title,
                    description=description,)

# Launch the demo!
demo.launch() 




