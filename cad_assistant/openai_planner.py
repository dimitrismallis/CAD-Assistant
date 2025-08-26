import os
import json
import base64
from io import BytesIO
from typing import Dict, Any, Optional
from openai import OpenAI
from PIL import Image
from .prompts import system_prompt, create_init_prompt, developer_prompt, create_followup_prompt

class OpenAIPlannerChain:
    """
    OpenAI-based planner for CAD-Assistant with minimal prompt system.
    """
    
    def __init__(self, config_path: str = "config.json", prompt_name: str = "default"):
        """
        Initialize the OpenAI planner.
        
        Args:
            config_path: Path to JSON config file containing API key and settings
            prompt_name: Name of the system prompt to use (must be loaded first)
        """
        self.config = self._load_config(config_path)
        self.client = OpenAI(api_key=self.config["openai_api_key"])
        self.model = self.config.get("openai_model", "gpt-4o-mini")
        self.prompt_name = prompt_name
        self.step_count = 0
        self.response_id = None
        
        # Image handling configuration
        self.logdir = None
    


    def _load_previous_step_image(self, previous_step) -> Optional[Dict[str, Any]]:
        """
        Load image from the previous step if it exists.
        
        Returns:
            Dictionary with image metadata and base64 data, or None if no image found
        """
        
        image_path = os.path.join(self.logdir, f"sketch_step_{previous_step}.png")
        
        if not os.path.exists(image_path):
            return None
        
        try:
            # Load and encode image
            with Image.open(image_path) as img:
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return {
                    "type": "sketch_render",
                    "data": base64_data,
                    "file_path": image_path,
                    "metadata": {
                        "step": previous_step,
                        "format": "PNG",
                        "size": img.size
                    }
                }
        except Exception as e:
            print(f"Warning: Could not load previous step image: {e}")
            return None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
        
    def __call__(self, context: Dict[str, Any], iteration_history) -> Dict[str, Any]:
        """
        Generate a single plan and action based on the current context.
        
        Args:
            context: Dictionary containing user_input, history, and env_init_response
            
        Returns:
            Dictionary with 'plan', 'action', and optionally 'image' keys
        """
        if self.step_count == 0:
            response = self._get_init_response(context)
            self.response_id = response.id
            content = response.output_text.strip()
            result = self._parse_response(content)
        else:
            structured_response = self._get_followup_response(iteration_history)
            
            # Extract text response
            response = structured_response["text_response"]
            self.response_id = response.id
            content = response.output_text.strip()
            result = self._parse_response(content)
            
            # Add image data if available
            if structured_response["image"]:
                result["image"] = structured_response["image"]
                result["metadata"] = structured_response["metadata"]
        
        # Increment step count for next call
        self.step_count += 1
        
        return result
            

    def _get_followup_response(self, iteration_history) -> Dict[str, Any]:
        # Load previous step image if available BEFORE making the request
        previous_step = len(iteration_history) - 1
        previous_image = self._load_previous_step_image(previous_step)
        
        # Build user message with automatic continuation
        user_message = create_followup_prompt(iteration_history)
        
        # Create message content - include image if available
        if previous_image:
            message_content = [
                {
                    "type": "input_text",
                    "text": user_message
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{previous_image['data']}"
                }
            ]
        else:
            message_content = user_message

        messages = [
            {"role": "user", "content": message_content}
        ]

        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                temperature=0.0,
                previous_response_id=self.response_id,
            )
        except:
            pass
        
        # Create structured response
        structured_response = {
            "text_response": response,
            "image": previous_image,
            "metadata": {
                "step": previous_step+1,
                "has_previous_image": previous_image is not None
            }
        }
        
        return structured_response

    def _get_init_response(self, context) -> str:

        user_input = context.get("user_input", "")

        env_init_response = context.get("env_init_response", "")

        # Get system prompt (uses system.j2 by default)
        system_message = system_prompt()
    
        # Get developer prompt sections config and pass to template
        freecad_documentation = self.config.get("freecad_documentation", {})

        # Build user message with automatic continuation
        user_message = create_init_prompt(user_input, self.step_count)

        # Get developer prompt with initialization code and section flags (uses developer.j2 by default) 
        developer_message = developer_prompt(
            initialcode=env_init_response,
            **freecad_documentation
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": developer_message},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                temperature= 0.0
            )
        except:
            pass
        
        return response


    def _parse_response(self, content: str) -> Dict[str, str]:
        """Parse the response to extract plan text and Python code blocks."""
        try:
            # Look for ACTION substring to split plan from action
            action_idx = content.find("ACTION")
            
            if action_idx != -1:
                # Plan is everything before ACTION
                plan = content[:action_idx].strip()
            else:
                # No ACTION found, entire content is plan
                plan = content.strip()
            
            # Extract Python code from ```python code blocks
            action = ""
            import re
            
            # Find all Python code blocks
            python_blocks = re.findall(r'```python\s*(.*?)\s*```', content, re.DOTALL)
            
            if python_blocks:
                # Join all Python code blocks with newlines
                action = '\n'.join(python_blocks).strip()
            
            return {
                "plan": plan,
                "action": action
            }
            
        except Exception as e:
            return {
                "plan": f"Failed to parse response: {str(e)}",
                "action": ""
            }
            
    def reset_steps(self):
        """Reset step counter."""
        self.step_count = 0 

