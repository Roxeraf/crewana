# streamlit_app.py
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import pandas as pd
import os
from dotenv import load_dotenv
from custom_tools import QualityDataAnalysisTool, ProcessDataAnalysisTool, DataVisualizationTool, OutlierDetectionTool

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in .env file or Streamlit secrets.")
    st.stop()

# Initialize OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Initialize tools
quality_tool = QualityDataAnalysisTool()
process_tool = ProcessDataAnalysisTool()
visualization_tool = DataVisualizationTool()
outlier_tool = OutlierDetectionTool()

# Define agents with specific tools
quality_analyst = Agent(
    role="Quality Analyst",
    goal="Analyze quality data to identify trends, issues, and improvement opportunities",
    backstory="You are an experienced quality analyst with expertise in statistical process control and quality management systems.",
    llm=llm,
    tools=[quality_tool, visualization_tool, outlier_tool]
)

process_analyst = Agent(
    role="Process Analyst",
    goal="Analyze process data to optimize production efficiency and identify bottlenecks",
    backstory="You have extensive experience in process engineering and lean manufacturing principles.",
    llm=llm,
    tools=[process_tool, visualization_tool]
)

data_scientist = Agent(
    role="Data Scientist",
    goal="Perform advanced analytics on combined quality and process data",
    backstory="You're an expert in machine learning and statistical analysis with a focus on manufacturing applications.",
    llm=llm,
    tools=[quality_tool, process_tool, visualization_tool, outlier_tool]
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
            # Convert dataframes to JSON strings for tool input
            quality_json = quality_df.to_json()
            process_json = process_df.to_json()

            # Define tasks
            quality_analysis = Task(
                description=f"Analyze the quality data focusing on {analysis_focus}. Use the Quality Data Analysis Tool and other available tools to support your analysis. Identify key quality metrics, trends, and potential issues. Quality data: {quality_json}",
                agent=quality_analyst
            )

            process_analysis = Task(
                description=f"Analyze the process data focusing on {analysis_focus}. Use the Process Data Analysis Tool and other available tools to support your analysis. Identify efficiency metrics, bottlenecks, and areas for improvement. Process data: {process_json}",
                agent=process_analyst
            )

            combined_analysis = Task(
                description=f"Perform advanced analytics on both quality and process data. Use all available tools to support your analysis. Identify correlations between process parameters and quality outcomes related to {analysis_focus}. Use insights from previous analyses. Quality data: {quality_json}, Process data: {process_json}",
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


