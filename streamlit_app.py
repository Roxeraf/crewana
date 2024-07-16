import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in .env file or Streamlit secrets.")
    st.stop()

# Initialize OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Define agents
process_engineer = Agent(
    role="Process Engineer",
    goal="Optimize the production process flow and efficiency",
    backstory="You have 15 years of experience in industrial process engineering and specialize in lean manufacturing.",
    llm=llm
)

data_scientist = Agent(
    role="Data Scientist",
    goal="Analyze production data to uncover patterns and insights",
    backstory="You're an expert in machine learning and statistical analysis with a focus on industrial applications.",
    llm=llm
)

data_analyst = Agent(
    role="Data Analyst",
    goal="Prepare and visualize production data for easy interpretation",
    backstory="You excel at transforming raw data into meaningful visualizations and reports.",
    llm=llm
)

programmer = Agent(
    role="Programmer",
    goal="Develop and maintain software tools for production monitoring and automation",
    backstory="You're a skilled software engineer with expertise in industrial automation and IoT.",
    llm=llm
)

quality_control = Agent(
    role="Quality Control Specialist",
    goal="Ensure product quality meets or exceeds standards throughout the production process",
    backstory="You have a keen eye for detail and deep knowledge of quality management systems.",
    llm=llm
)

report_writer = Agent(
    role="Report Writer",
    goal="Compile all findings and recommendations into a comprehensive, well-structured report",
    backstory="You're a skilled technical writer with experience in creating clear, concise reports for complex industrial processes.",
    llm=llm
)

# Streamlit app
st.title("Production Process Analysis")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your production data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # User input for specific analysis
    analysis_focus = st.text_input("What aspect of the production process would you like to analyze?")

    if st.button("Analyze Process"):
        if analysis_focus:
            # Define tasks
            data_preparation = Task(
                description=f"Prepare and clean the production data for analysis, focusing on {analysis_focus}. Provide a summary of the data preparation steps.",
                agent=data_analyst
            )

            data_analysis = Task(
                description=f"Analyze the prepared data to identify patterns and insights related to {analysis_focus}. Provide detailed findings.",
                agent=data_scientist
            )

            process_optimization = Task(
                description=f"Based on the data analysis, suggest process improvements for {analysis_focus}. Provide specific, actionable recommendations.",
                agent=process_engineer
            )

            quality_assessment = Task(
                description=f"Evaluate the quality implications of the proposed improvements for {analysis_focus}. Discuss potential risks and mitigation strategies.",
                agent=quality_control
            )

            automation_suggestion = Task(
                description=f"Propose software solutions or automations to support the improvements in {analysis_focus}. Include high-level implementation steps.",
                agent=programmer
            )

            final_report = Task(
                description=f"Compile a comprehensive report on the analysis of {analysis_focus} in the production process. Include an executive summary, detailed findings from each specialist, recommendations, and next steps.",
                agent=report_writer
            )

            # Create Crew
            crew = Crew(
                agents=[process_engineer, data_scientist, data_analyst, programmer, quality_control, report_writer],
                tasks=[data_preparation, data_analysis, process_optimization, quality_assessment, automation_suggestion, final_report],
                verbose=True
            )

            # Execute the crew's tasks
            with st.spinner("Analyzing process and generating report... This may take a few minutes."):
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
                file_name="production_process_analysis_report.md",
                mime="text/markdown"
            )
        else:
            st.warning("Please specify an aspect of the production process to analyze.")

else:
    st.info("Please upload a CSV file containing your production data.")

# Additional app sections
st.subheader("About This Tool")
st.write("""
This tool uses AI agents powered by GPT-3.5 to analyze your production process data. 
It combines the expertise of a Process Engineer, Data Scientist, Data Analyst, 
Programmer, and Quality Control Specialist to provide comprehensive insights 
and recommendations for process improvement. A final report is generated to summarize all findings.
""")

st.subheader("How to Use")
st.write("""
1. Upload your production data CSV file.
2. Specify the aspect of the production process you want to analyze.
3. Click 'Analyze Process' to get insights, recommendations, and a full report.
4. Download the report for offline viewing or sharing.
""")