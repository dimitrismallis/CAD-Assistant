"""
Minimal Prompt System with J2 Files
===================================

Ultra-simple prompt system loading from .j2 template files:
- system_prompt() - loads from system.j2
- developer_prompt() - loads from developer.j2  
- user_followup() - loads from followup.j2
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader

# Get the directory containing prompt templates
PROMPT_DIR = Path(__file__).parent / "prompts"

# Initialize Jinja2 environment
jinja_env = Environment(
    loader=FileSystemLoader(PROMPT_DIR),
    trim_blocks=True,
    lstrip_blocks=True
)

def system_prompt(template_file: str = "system", **variables) -> Optional[str]:
    """
    Load and render system prompt from {template_file}.j2 file.
    
    Default loads from system.j2:
    prompt = system_prompt(assistant_name='CAD-Expert', domain='modeling')
    
    Custom file:
    prompt = system_prompt('custom_system', assistant_name='CAD-Expert')
    """
    try:
        template = jinja_env.get_template(f"{template_file}.j2")
        return template.render(**variables)
    except Exception as e:
        print(f"❌ Failed to load system prompt '{template_file}.j2': {e}")
        return None

def developer_prompt(template_file: str = "developer", **variables) -> Optional[str]:
    """
    Load and render developer prompt from {template_file}.j2 file.
    
    Default loads from developer.j2:
    prompt = developer_prompt(mode='verbose', debug_level='high')
    
    Custom file:
    prompt = developer_prompt('custom_debug', mode='verbose')
    """
    try:
        template = jinja_env.get_template(f"{template_file}.j2")
        
        #load freecad documentation
        freecad_doc_template = jinja_env.get_template(f"freecad_doc.j2")
        freecad_doc = freecad_doc_template.render(**variables)
        variables['freecad_doc'] = freecad_doc

        # Load tool documentation and append to the prompt
        tool_docs = _load_tool_documentation()
        if tool_docs:
            variables['tool_documentation'] = tool_docs
        
        return template.render(**variables)
    except Exception as e:
        print(f"❌ Failed to load developer prompt '{template_file}.j2': {e}")
        return None

def _load_tool_documentation() -> str:
    """Load tool documentation from cad_tools directory based on config."""
    import os
    import json
    
    # Load config to get registered tools
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if not os.path.exists(config_path):
        return ""
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return ""
    
    registered_tools = config.get("tools", [])
    if not registered_tools:
        return ""
    
    tools_dir = os.path.join(os.path.dirname(__file__), "cad_tools")
    if not os.path.exists(tools_dir):
        return ""
    
    tool_docs = []
    
    # Load documentation for each registered tool
    for tool_name in registered_tools:
        doc_file_path = os.path.join(tools_dir, f"{tool_name}_doc.txt")
        
        try:
            if os.path.exists(doc_file_path):
                with open(doc_file_path, 'r') as f:
                    doc_content = f.read()
                tool_docs.append(doc_content)
            else:
                print(f"⚠️ Documentation file not found for tool: {tool_name}")
        except Exception as e:
            print(f"❌ Failed to load documentation for {tool_name}: {e}")
    
    if tool_docs:
        return '\n\n'.join(tool_docs)
    return ""

def create_init_prompt(user_input: str, history: list, step_count: int = 0, template_file: str = "init_prompt") -> str:
    """
    Load and render init prompt from {template_file}.j2 file.
    
    """
    try:
        template = jinja_env.get_template(f"{template_file}.j2")
        return template.render(
            user_input=user_input,
        )
    except Exception as e:
        print(f"❌ Failed to load followup template '{template_file}.j2': {e}")

def create_followup_prompt(iteration_history, template_file: str = "followup_prompt") -> str:
    """
    Load and render followup prompt from {template_file}.j2 file.
    
    Args:
        context: Dictionary containing exit_code, output, and other variables for the template
        template_file: Name of the template file (without .j2 extension)
    
    Returns:
        Rendered template string
    """
    try:
        template = jinja_env.get_template(f"{template_file}.j2")
        return template.render(**iteration_history[-1])
    except Exception as e:
        print(f"❌ Failed to load followup template '{template_file}.j2': {e}")
        return ""

# Export only the essentials
__all__ = ['system_prompt', 'developer_prompt', 'create_init_prompt', 'create_followup_prompt'] 