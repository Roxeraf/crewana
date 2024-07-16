import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in .env file or Streamlit secrets.")
    st.stop()

# Initialize OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Define tools
def calculate_statistics(data):
    return data.describe().to_string()

def create_correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def identify_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    return outliers.to_string()

# Define agents with tools
quality_analyst = Agent(
    role="Quality Analyst",
    goal="Analyze quality data to identify trends, issues, and improvement opportunities",
    backstory="You are an experienced quality analyst with expertise in statistical process control and quality management systems.",
    llm=llm,
    tools=[
        Tool(name="Calculate Statistics", func=calculate_statistics, description="Calculate basic statistics of the data"),
        Tool(name="Identify Outliers", func=identify_outliers, description="Identify outliers in the data")
    ]
)

process_analyst = Agent(
    role="Process Analyst",
    goal="Analyze process data to optimize production efficiency and identify bottlenecks",
    backstory="You have extensive experience in process engineering and lean manufacturing principles.",
    llm=llm,
    tools=[
        Tool(name="Calculate Statistics", func=calculate_statistics, description="Calculate basic statistics of the data"),
        Tool(name="Create Correlation Heatmap", func=create_correlation_heatmap, description="Create a correlation heatmap of the data")
    ]
)

data_scientist = Agent(
    role="Data Scientist",
    goal="Perform advanced analytics on combined quality and process data",
    backstory="You're an expert in machine learning and statistical analysis with a focus on manufacturing applications.",
    llm=llm,
    tools=[
        Tool(name="Calculate Statistics", func=calculate_statistics, description="Calculate basic statistics of the data"),
        Tool(name="Create Correlation Heatmap", func=create_correlation_heatmap, description="Create a correlation heatmap of the data"),
        Tool(name="Identify Outliers", func=identify_outliers, description="Identify outliers in the data")
    ]
)

report_writer = Agent(
    role="Report Writer",
    goal="Compile all findings and recommendations into a comprehensive, actionable report",
    backstory="You're a skilled technical writer with experience in creating clear, concise reports for manufacturing environments.",
    llm=llm
)

# Streamlit app
st.title("Quality and Process Data Analysis")

# File uploaders for CSV
quality_file = st.file_uploader("Upload your quality data (CSV)", type="csv")
process_file = st.file_uploader("Upload your process data (CSV)", type="csv")

if quality_file is not None and process_file is not None:
    quality_df = pd.read_csv(quality_file)
    process_df = pd.read_csv(process_file)
    
    st.subheader("Quality Data Preview")
    st.write(quality_df.head())
    
    st.subheader("Process Data Preview")
    st.write(process_df.head())

    # User input for specific analysis focus
    analysis_focus = st.text_input("What specific aspect of quality or process would you like to focus on?")

    if st.button("Analyze Data"):
        if analysis_focus:
            # Define tasks
            quality_analysis = Task(
                description=f"Analyze the quality data focusing on {analysis_focus}. Use the Calculate Statistics and Identify Outliers tools to support your analysis. Identify key quality metrics, trends, and potential issues.",
                agent=quality_analyst
            )

            process_analysis = Task(
                description=f"Analyze the process data focusing on {analysis_focus}. Use the Calculate Statistics and Create Correlation Heatmap tools to support your analysis. Identify efficiency metrics, bottlenecks, and areas for improvement.",
                agent=process_analyst
            )

            combined_analysis = Task(
                description=f"Perform advanced analytics on both quality and process data. Use all available tools to support your analysis. Identify correlations between process parameters and quality outcomes related to {analysis_focus}. Use insights from previous analyses.",
                agent=data_scientist
            )

            final_report = Task(
                description=f"Compile a comprehensive report on the analysis of {analysis_focus}. Include key findings from quality and process analyses, advanced insights, recommendations for improvement, and suggested next steps.",
                agent=report_writer
            )

            # Create Crew
            crew = Crew(
                agents=[quality_analyst, process_analyst, data_scientist, report_writer],
                tasks=[quality_analysis, process_analysis, combined_analysis, final_report],
                verbose=True
            )

            # Execute the crew's tasks
            with st.spinner("Analyzing data and generating report... This may take a few minutes."):
                result = crew.kickoff()

            # Display results
            st.subheader("Analysis Summary")
            st.write(result)

            st.subheader("Full Report")
            report = crew.tasks[-1].output  # Get the output of the final task (report writing)
            st.markdown(report)

            # Option to download the report
            st.download_button(
                label="Download Full Report",
                data=report,
                file_name="quality_process_analysis_report.md",
                mime="text/markdown"
            )
        else:
            st.warning("Please specify an aspect of quality or process to analyze.")

else:
    st.info("Please upload both quality and process data CSV files to begin the analysis.")

# Additional app sections (About This Tool and How to Use) remain the same