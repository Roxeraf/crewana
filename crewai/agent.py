def _parse_tools(self, tools):
    tools_list = []
    try:
        from crewai_tools import BaseTool as CrewAITool
        for tool in tools:
            if isinstance(tool, CrewAITool):
                tools_list.append(tool)
    except ModuleNotFoundError:
        pass  # Handle the case where crewai_tools is not available

    for tool in tools:
        tools_list.append(tool)  # Add custom tools or handle them in another way

    return tools_list
