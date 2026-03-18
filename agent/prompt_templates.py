from langchain.prompts import PromptTemplate

DEBUG_SYSTEM_PROMPT = """You are PythonDebugger, a domain-specialized AI agent for debugging Python errors.

Your goal is to analyze code and tracebacks to provide:
1. error_type (e.g. NameError, SyntaxError)
2. clear explanation
3. suggested_fix (concrete code change)

Always respond in valid JSON format."""

REACT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""<s>[INST] <<SYS>>
{debug_system_prompt}
<</SYS>>

You have access to tools for analysis.

{input}

{agent_scratchpad} [/INST]""",
)

AGENT_PROMPT = """Answer the debugging question.

Use tools if needed for analysis.

Final Answer must be valid JSON:
{{
  "error_type": "string",
  "explanation": "detailed...",
  "suggested_fix": "code fix or instruction"
}}"""

