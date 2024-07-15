import sqlite3
print(f"SQLite version: {sqlite3.sqlite_version}")

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Try to get the API key from environment variables or Streamlit secrets
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ API key not found. Please set it in .env file or Streamlit secrets.")
    st.stop()

# Set up Groq client
groq_model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8k")

# Define your agents
data_analyst = Agent(
    role='Data Analyst',
    goal='Analyze data and provide insights',
    backstory='You are an expert data analyst with years of experience in various industries.',
    llm=groq_model
)

data_visualizer = Agent(
    role='Data Visualizer',
    goal='Create clear and insightful visualizations',
    backstory='You are a skilled data visualizer with expertise in creating impactful charts and graphs.',
    llm=groq_model
)

# Streamlit app
st.title("AI-Powered Data Analysis Assistant")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write(df)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Define tasks with user input
            analysis_task = Task(
                description=f'Analyze the provided dataset and extract key insights related to: {prompt}',
                agent=data_analyst,
                expected_output="A detailed report of the data analysis findings"
            )

            visualization_task = Task(
                description=f'Create visualizations based on the analysis results for the query: {prompt}',
                agent=data_visualizer,
                expected_output="A list of visualization suggestions with descriptions"
            )

            # Create the crew
            data_crew = Crew(
                agents=[data_analyst, data_visualizer],
                tasks=[analysis_task, visualization_task]
            )
            
            # Generate response using crewAI
            result = data_crew.kickoff()
            
            # Simulate stream of response with milliseconds delay
            for chunk in result.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add data summary and basic stats
if uploaded_file is not None:
    st.subheader("Data Summary")
    st.write(df.describe())
    
    st.subheader("Column Information")
    st.write(df.dtypes)

# Add more Streamlit components and functionality as needed