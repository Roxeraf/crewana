import streamlit as st
from crewai import Task, Crew
from dotenv import load_dotenv
from crewai.agents import quality_analyst, process_analyst, data_scientist, report_writer
import pandas as pd
import os

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in .env file or Streamlit secrets.")
    st.stop()

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
            report = crew.tasks[-1].output  # Get




