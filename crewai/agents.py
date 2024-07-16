# crewai/agents.py
from crewai import Agent
from langchain_openai import ChatOpenAI
from .tools import QualityDataAnalysisTool, ProcessDataAnalysisTool, DataVisualizationTool, OutlierDetectionTool

# Initialize OpenAI model (you can adjust this as needed)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="YOUR_API_KEY")

# Initialize tools
quality_tool = QualityDataAnalysisTool()
process_tool = ProcessDataAnalysisTool()
visualization_tool = DataVisualizationTool()
outlier_tool = OutlierDetectionTool()

# Create a list of tools to pass to agents
quality_tools = [quality_tool, visualization_tool, outlier_tool]
process_tools = [process_tool, visualization_tool]
data_scientist_tools = [quality_tool, process_tool, visualization_tool, outlier_tool]

# Define agents with specific tools
quality_analyst = Agent(
    role="Quality Analyst",
    goal="Analyze quality data to identify trends, issues, and improvement opportunities",
    backstory="You are an experienced quality analyst with expertise in statistical process control and quality management systems.",
    llm=llm,
    tools=quality_tools
)

process_analyst = Agent(
    role="Process Analyst",
    goal="Analyze process data to optimize production efficiency and identify bottlenecks",
    backstory="You have extensive experience in process engineering and lean manufacturing principles.",
    llm=llm,
    tools=process_tools
)

data_scientist = Agent(
    role="Data Scientist",
    goal="Perform advanced analytics on combined quality and process data",
    backstory="You're an expert in machine learning and statistical analysis with a focus on manufacturing applications.",
    llm=llm,
    tools=data_scientist_tools
)

report_writer = Agent(
    role="Report Writer",
    goal="Compile all findings and recommendations into a comprehensive, actionable report",
    backstory="You're a skilled technical writer with experience in creating clear, concise reports for manufacturing environments.",
    llm=llm
)
