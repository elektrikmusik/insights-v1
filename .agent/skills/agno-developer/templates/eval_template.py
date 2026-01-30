import asyncio
from agno.agent import Agent
from agno.models.google import Gemini
from src.config import settings
from typing import List, Dict

# Import the agent you want to evaluate
# from src.agents.your_agent import YourAgent

class AgentEvaluator:
    """
    Evaluator for Agno Agents using LLM-as-a-judge.
    """
    
    def __init__(self):
        self.judge = Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=settings.gemini_api_key),
            description="Evaluation Judge",
            instructions="You are an improper evaluation judge. Compare the Actual Output against the Expected Criteria.",
        )

    async def evaluate_response(self, input_text: str, actual_output: str, criteria: str) -> dict:
        prompt = f"""Evaluate the following agent response.

Input: {input_text}

Actual Output:
{actual_output}

Evaluation Criteria:
{criteria}

Score (0-10) and Reasoning:
"""
        response = await self.judge.arun(prompt)
        return {"input": input_text, "evaluation": response.content}

async def run_evals():
    # 1. Setup Agent
    # agent = YourAgent()
    
    # 2. Define Test Cases
    test_cases = [
        {"input": "Test Input 1", "criteria": "Should mention X and Y"},
        {"input": "Test Input 2", "criteria": "Should not mention Z"},
    ]
    
    # 3. Run Loop
    evaluator = AgentEvaluator()
    results = []
    
    for case in test_cases:
        print(f"Running eval for: {case['input']}")
        # actual = await agent.run(case['input'])
        actual = "Placeholder output" # Replace with actual agent run
        
        eval_result = await evaluator.evaluate_response(case['input'], actual, case['criteria'])
        results.append(eval_result)
        print(f"Result: {eval_result['evaluation']}\n")

if __name__ == "__main__":
    asyncio.run(run_evals())
