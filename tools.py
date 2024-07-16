# tools.py
from langchain.tools import Tool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

def calculate_statistics(data: pd.DataFrame) -> str:
    return data.describe().to_string()

def create_correlation_heatmap(data: pd.DataFrame) -> bytes:
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()

def identify_outliers(data: pd.DataFrame) -> str:
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    return outliers.to_string()

tools = [
    Tool(name="Calculate Statistics", func=calculate_statistics, description="Calculate basic statistics of the data"),
    Tool(name="Create Correlation Heatmap", func=create_correlation_heatmap, description="Create a correlation heatmap of the data"),
    Tool(name="Identify Outliers", func=identify_outliers, description="Identify outliers in the data")
]