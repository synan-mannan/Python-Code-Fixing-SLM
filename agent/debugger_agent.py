import json
import torch
from typing import Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from .prompt_templates import REACT_PROMPT_TEMPLATE, DEBUG_SYSTEM_PROMPT
from .tools import ast_parser, static_analyzer, error_classifier

class PythonDebuggerAgent:
    def __init__(self, model_path: str = "models/slm-debugger"):
        """
        Initialize the debugging agent with fine-tuned SLM.
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.tokenizer.model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        # LLM Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        self.tools = [
            Tool(
                name="AST_Parser",
                func=ast_parser,
                description="useful for syntax checking and structure analysis"
            ),
            Tool(
                name="Static_Analyzer",
                func=static_analyzer,
                description="detects common static errors and patterns"
            ),
            Tool(
                name="Error_Classifier",
                func=error_classifier,
                description="quickly classify error type from traceback"
            ),
        ]
        
        # React Agent
        self.prompt = REACT_PROMPT_TEMPLATE.partial(debug_system_prompt=DEBUG_SYSTEM_PROMPT)
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True
        )
    
    def debug(self, code: str, traceback: str = "") -> Dict[str, str]:
        """
        Debug Python code + traceback.
        
        Returns structured JSON.
        """
        input_prompt = f"""
Code:
```{code}```

Traceback:
```{traceback}```

Debug this error.
        """
        
        try:
            result = self.agent_executor.invoke({"input": input_prompt})
            # Parse JSON from final answer
            output = result["output"]
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "error_type": parsed.get("error_type", "Unknown"),
                    "explanation": parsed.get("explanation", ""),
                    "suggested_fix": parsed.get("suggested_fix", "")
                }
            else:
                return {
                    "error_type": "Unknown",
                    "explanation": output,
                    "suggested_fix": "Could not parse suggestion"
                }
        except Exception as e:
            return {
                "error_type": "AgentError",
                "explanation": str(e),
                "suggested_fix": ""
            }

if __name__ == "__main__":
    agent = PythonDebuggerAgent()
    result = agent.debug(
        code="print(undefined_var)",
        traceback="NameError: name 'undefined_var' is not defined"
    )
    print(json.dumps(result, indent=2))

