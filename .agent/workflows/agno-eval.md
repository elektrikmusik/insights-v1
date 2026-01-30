---
description: Evaluate an Agno agent using LLM-as-a-judge.
---

# Evaluate Agno Agent

1.  **Setup Evaluation Script**:
    - Create a new script in `backend/scripts/` (e.g., `eval_[agent_name].py`).
    - Copy the content from `.agent/skills/agno-developer/templates/eval_template.py`.
2.  **Configure**:
    - Import the specific agent you want to test.
    - Define `test_cases` with inputs and success criteria.
3.  **Run Evaluation**:
    - Run: `python backend/scripts/eval_[agent_name].py`
4.  **Analyze**:
    - Review the "Score and Reasoning" output from the judge.
    - Iterate on the agent's instructions or tools to improve the score.
