
import pandas as pd
import matplotlib.pyplot as plt

def plot_label_counts(df: pd.DataFrame):
    ax = df['label'].value_counts().sort_values(ascending=False).plot(kind='bar', title='Crop label counts')
    plt.tight_layout()
    plt.show()
