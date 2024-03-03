from FilePaths import FilePaths
import pandas as pd
import numpy as np
from tabulate import tabulate
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt
init_notebook_mode(connected=True)
cf.go_offline()

# Function to count words in an essay
def count_words(essay):
    words = essay.split()
    return len(words)