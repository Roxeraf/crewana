# custom_tools.py
from crewai_tools import BaseTool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

class QualityDataAnalysisTool(BaseTool):
    name: str = "Quality Data Analysis Tool"
    description: str = "Analyzes quality data to identify trends, issues, and improvement opportunities."

    def _run(self, data: str) -> str:
        df = pd.read_json(data)
        stats = df.describe().to_string()
        return f"Quality Data Analysis:\n{stats}"

class ProcessDataAnalysisTool(BaseTool):
    name: str = "Process Data Analysis Tool"
    description: str = "Analyzes process data to optimize production efficiency and identify bottlenecks."

    def _run(self, data: str) -> str:
        df = pd.read_json(data)
        efficiency = df['efficiency'].mean() if 'efficiency' in df.columns else "N/A"
        bottlenecks = df.min().to_string()
        return f"Process Efficiency: {efficiency}\nPotential Bottlenecks:\n{bottlenecks}"

class DataVisualizationTool(BaseTool):
    name: str = "Data Visualization Tool"
    description: str = "Creates visualizations of data for better insights."

    def _run(self, data: str) -> str:
        df = pd.read_json(data)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return "Correlation heatmap created successfully. (Image data not shown in text output)"

class OutlierDetectionTool(BaseTool):
    name: str = "Outlier Detection Tool"
    description: str = "Identifies outliers in the dataset."

    def _run(self, data: str) -> str:
        df = pd.read_json(data)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
        return f"Outliers detected:\n{outliers.to_string()}"