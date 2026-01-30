from agno.agent import Agent
from agno.models.google import Gemini
from src.config import settings

class TemplateAgent:
    """
    Template for creating a specific agent role within the Agno framework.
    """
    
    def __init__(self):
        # Initialize the Agno Agent with Gemini model
        self.agent = Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=settings.gemini_api_key),
            description="[Agent Role Description]",
            instructions="""You are an expert [ROLE].
            
## Your Mission:
- [Goal 1]
- [Goal 2]

## Your Instructions:
1. [Instruction 1]
2. [Instruction 2]

## Output Format:
- [Format details]
""",
            markdown=True,
            # tools=[self.my_tool_function], # Optional: Add tools here
            # storage=... # Optional: Add storage here
            # knowledge=... # Optional: Add knowledge base here
        )

    async def run_task(self, inputs: dict) -> dict:
        """
        Execute the agent's primary task with the given inputs.
        """
        prompt = f"""Task: [Describe the task using inputs]
        
Inputs:
{inputs}
"""
        response = await self.agent.arun(prompt)
        
        return {
            "result": response.content,
            "status": "success"
        }
