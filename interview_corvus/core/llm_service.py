from typing import List, Type, TypeVar
import re  # Add the missing import for regular expressions

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from loguru import logger
from pydantic import BaseModel
from PyQt6.QtCore import QObject, pyqtSignal

from interview_corvus.config import settings
from interview_corvus.core.models import CodeOptimization, CodeSolution
from interview_corvus.core.prompt_manager import PromptManager
from interview_corvus.security.api_key_manager import APIKeyManager

T = TypeVar("T", bound=BaseModel)


class CustomOpenAI(OpenAI):
    """Custom OpenAI class with overridden metadata property for OpenRouter models."""
    
    def __init__(self, **kwargs):
        """Initialize with custom tokenizer for OpenRouter models."""
        super().__init__(**kwargs)
        
        # Pre-load tokenizer based on model
        self._custom_tokenizer = None
        try:
            import tiktoken
            # For NVIDIA Llama models ALWAYS use cl100k_base 
            if "nvidia" in self.model and "nemotron" in self.model:
                # Using cl100k_base as it's compatible with many modern models
                self._custom_tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info(f"Preloaded cl100k_base tokenizer for {self.model}")
            # For other models
            else:
                self._custom_tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to preload tokenizer: {e}")
            
        # Flag to completely bypass tokenization for problematic models
        self._bypass_tokenization = "nvidia" in self.model and "nemotron" in self.model
        if self._bypass_tokenization:
            logger.info(f"Set to bypass tokenization for model: {self.model}")
    
    @property
    def metadata(self):
        """Return metadata with correct context window sizes for OpenRouter models."""
        model_name = self.model
        
        if "nvidia" in model_name and "nemotron-ultra" in model_name:
            # According to docs, Llama 3.1 Nemotron Ultra 253B has 128K context
            return LLMMetadata(
                context_window=131072,  # 128K tokens
                model_name=model_name,
                num_output=4096,  # Default max output tokens
            )
        elif "deepseek" in model_name:
            # DeepSeek models typically have 32K context
            return LLMMetadata(
                context_window=32768,  # 32K tokens
                model_name=model_name,
                num_output=4096,  # Default max output tokens
            )
        
        # For standard OpenAI models, use parent implementation
        return super().metadata
        
    def _get_tokenizer(self):
        """Override tokenizer method to handle OpenRouter models."""
        # If we have a preloaded tokenizer, use it
        if self._custom_tokenizer is not None:
            return self._custom_tokenizer
            
        try:
            # First try the default tokenizer approach
            return super()._get_tokenizer()
        except Exception as e:
            # If that fails (e.g., for OpenRouter models), use a fallback approach
            logger.info(f"Using fallback tokenizer for model: {self.model}")
            
            # Import tiktoken here to avoid circular imports
            import tiktoken
            
            # For NVIDIA Llama models, use cl100k_base which works well with many Llama models
            if "nvidia" in self.model and "nemotron" in self.model:
                logger.info("Using cl100k_base tokenizer for NVIDIA Llama model")
                self._custom_tokenizer = tiktoken.get_encoding("cl100k_base")
                return self._custom_tokenizer
            
            # For other OpenRouter models, try to use a suitable tokenizer
            try:
                self._custom_tokenizer = tiktoken.get_encoding("cl100k_base")  # Default to cl100k_base as it's widely compatible
                return self._custom_tokenizer
            except:
                # If all else fails, fall back to the most basic tokenizer
                self._custom_tokenizer = tiktoken.get_encoding("gpt2")
                return self._custom_tokenizer
    
    def encode(self, text):
        """Override encode to handle tokenization issues."""
        try:
            # Try to use the default encoding method
            tokenizer = self._get_tokenizer()
            return tokenizer.encode(text)
        except Exception as e:
            logger.warning(f"Encoding failed, using fallback method: {e}")
            # Use a simple character-based tokenization as absolute fallback
            # This is not accurate but prevents crashing
            return [ord(c) for c in text[:1000]]  # Limit to 1000 chars for safety
            
    def get_token_count(self, text):
        """Get the number of tokens in the text."""
        # For NVIDIA models, use our fixed tokenizer
        if self._bypass_tokenization:
            try:
                if self._custom_tokenizer:
                    return len(self._custom_tokenizer.encode(text))
                else:
                    # Very rough approximation
                    return len(text) // 4
            except Exception as e:
                logger.warning(f"Token count fallback for NVIDIA: {e}")
                return 100  # Just return a reasonable number
                
        # Otherwise try regular approaches
        try:
            return super().get_token_count(text)
        except Exception as e:
            logger.info(f"Using fallback token counting for model: {self.model}")
            try:
                tokenizer = self._get_tokenizer()
                return len(tokenizer.encode(text))
            except Exception as inner_e:
                logger.warning(f"All tokenization methods failed: {inner_e}")
                return len(text) // 4  # Rough approximation
                
    def tokenize(self, text):
        """Override tokenize method to handle issues with OpenRouter models."""
        # For NVIDIA models, always use our fixed approach
        if self._bypass_tokenization:
            try:
                if self._custom_tokenizer:
                    return self._custom_tokenizer.encode(text)
                else:
                    # If we failed to load a tokenizer, just return a minimalist version
                    # This won't be used for actual API calls, just for token counting
                    return [ord(c) for c in text[:100]]  # Just return a few tokens
            except Exception as e:
                logger.warning(f"Tokenizer fallback failed: {e}")
                return [0] * 10  # Absolute minimal fallback
                
        # Otherwise try the standard approach
        try:
            if hasattr(super(), "tokenize"):
                return super().tokenize(text)
        except Exception as e:
            logger.warning(f"Default tokenize failed: {e}")
            
        # Ultimate fallback
        try:
            return self.encode(text)
        except Exception as e:
            logger.warning(f"Tokenize fallback failed: {e}")
            return [0] * 10  # Absolute minimal fallback


class LLMService(QObject):
    """Service for interacting with LLM through LlamaIndex."""

    # Signals for responses
    completion_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        """Initialize the LLM service with configured settings."""
        super().__init__()
        api_key = APIKeyManager().get_api_key()

        # Determine if we're using OpenAI or Anthropic based on model name
        is_anthropic = any(
            model_prefix in settings.llm.model
            for model_prefix in ["claude", "anthropic"]
        )

        if is_anthropic:
            self.llm = Anthropic(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                api_key=api_key,
                max_tokens=12000,  # Set an appropriate max tokens value
            )
        else:
            # Check if using OpenRouter based on model name or setting
            use_openrouter = settings.llm.use_openrouter or any(
                prefix in settings.llm.model 
                for prefix in ["deepseek", "nvidia"]
            )
            
            # Initialize OpenAI client with OpenRouter support
            if use_openrouter:
                openai_kwargs = {
                    "model": settings.llm.model,
                    "temperature": settings.llm.temperature,
                    "api_key": api_key,
                    "max_retries": settings.llm.max_retries,
                    "timeout": settings.llm.timeout,
                    "base_url": settings.llm.openrouter_base_url,
                    "default_headers": {
                        "HTTP-Referer": settings.llm.openrouter_site_url,
                        "X-Title": settings.llm.openrouter_site_name,
                    }
                }
                logger.info(f"Initialized OpenRouter LLM with model: {settings.llm.model}")
                
                # Use our custom class for OpenRouter models
                self.llm = CustomOpenAI(**openai_kwargs)
            else:
                # Standard OpenAI configuration
                openai_kwargs = {
                    "model": settings.llm.model,
                    "temperature": settings.llm.temperature,
                    "api_key": api_key,
                    "max_retries": settings.llm.max_retries,
                    "timeout": settings.llm.timeout,
                }
                logger.info(f"Initialized OpenAI LLM with model: {settings.llm.model}")
                self.llm = OpenAI(**openai_kwargs)

        # Initialize chat engine that will maintain conversation history
        self.chat_engine = SimpleChatEngine.from_defaults(
            llm=self.llm,
        )

    def reset_chat_history(self):
        """Reset the chat history."""
        logger.info("Resetting chat history")
        # Recreate chat engine to clear the history
        self.chat_engine = SimpleChatEngine.from_defaults(
            llm=self.llm,
        )

    def get_code_optimization(
        self, code: str, language: str = None
    ) -> CodeOptimization:
        """
        Get an optimized version of provided code.

        Args:
            code: The code to optimize
            language: The programming language of the code (defaults to settings)

        Returns:
            A structured code optimization response
        """
        # Use default language from settings if none provided
        if language is None:
            language = settings.default_language

        prompt_manager = PromptManager()
        prompt = prompt_manager.get_prompt(
            "code_optimization", code=code, language=language
        )

        message = ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        )

        structured = self.llm.as_structured_llm(output_cls=CodeOptimization)
        response = structured.chat([message])
        return response.raw

    def get_solution_from_screenshots(
        self, screenshot_paths: List[str], language: str = None
    ) -> CodeSolution:
        """
        Get a solution based on multiple screenshots of a programming problem.

        Args:
            screenshot_paths: List of paths to the screenshot files
            language: The programming language to use for the solution (defaults to settings)

        Returns:
            A structured code solution response
        """
        # Use default language from settings if none provided
        if language is None:
            language = settings.default_language

        # For multimodal requests, using direct OpenAI API call through LlamaIndex
        prompt_manager = PromptManager()

        # Get the screenshot prompt with language
        prompt_text = prompt_manager.get_prompt(
            "screenshot_solution", language=language
        )

        logger.info(f"Processing {len(screenshot_paths)} screenshots")

        # Create a system message with the prompt text
        system = ChatMessage(
            role=MessageRole.SYSTEM,
            content=prompt_text,
        )

        # Initialize chat messages with the system message
        chat_messages = [system]

        # Add each screenshot as a user message with an image block
        for path in screenshot_paths:
            logger.info(f"Adding screenshot: {path}")
            screenshot = ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    ImageBlock(
                        path=path,
                    )
                ],
            )
            chat_messages.append(screenshot)

        # For processing screenshots with history context
        structured = self.llm.as_structured_llm(output_cls=CodeSolution)
        response = structured.chat(chat_messages)
        return response.raw

    def get_code_solution(
        self, problem_description: str, language: str = None
    ) -> CodeSolution:
        """
        Get a solution for a programming problem described in text.

        Args:
            problem_description: The description of the programming problem
            language: The programming language to use for the solution (defaults to settings)

        Returns:
            A structured code solution response
        """
        # Use default language from settings if none provided
        if language is None:
            language = settings.default_language
            
        # =====================================================================
        # SPECIAL CASE: DIRECT API HANDLING FOR NVIDIA LLAMA MODEL
        # =====================================================================
        # This completely bypasses LlamaIndex to avoid tokenization issues
        if "nvidia" in settings.llm.model and "nemotron" in settings.llm.model:
            logger.info(f"Using dedicated direct API path for NVIDIA model")
            return self._get_nvidia_solution_direct(problem_description, language)
            
        # Standard approach for all other models
        # Prepare prompt
        prompt_manager = PromptManager()
        prompt = prompt_manager.get_prompt(
            "code_solution", problem_description=problem_description, language=language
        )

        message = ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        )

        logger.info(f"Generating code solution in {language} for text problem")
        
        try:
            # Create a structured LLM for CodeSolution
            structured = self.llm.as_structured_llm(output_cls=CodeSolution)
            response = structured.chat([message])
            return response.raw
        except Exception as e:
            logger.error(f"Error generating solution from text: {e}")
            
            # Create a simple solution object with the error message
            return CodeSolution(
                code=f"# Error generating solution\n# {str(e)}\n\n# Please try a different model or approach",
                explanation=f"Error occurred during generation: {str(e)}",
                time_complexity="N/A",
                space_complexity="N/A",
                edge_cases=["N/A"],
                language=language or settings.default_language
            )
            
    def _get_nvidia_solution_direct(self, problem_description: str, language: str) -> CodeSolution:
        """Special dedicated handler for NVIDIA model that uses direct API and simple response handling"""
        try:
            # Get the OpenRouter API key
            api_key = APIKeyManager().get_api_key()
            
            # Prepare a very simple prompt that won't cause formatting issues
            simple_prompt = f"""Please solve this programming problem in {language}:

{problem_description}

Provide just the code solution, time complexity (O notation), space complexity (O notation), and a brief explanation.
"""
            logger.info(f"Using simplified direct API approach for NVIDIA model")
            
            import requests
            import json
            import re
            
            # Set up headers for OpenRouter API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": settings.llm.openrouter_site_url,
                "X-Title": settings.llm.openrouter_site_name
            }
            
            # Simple request with minimal parameters
            data = {
                "model": settings.llm.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": simple_prompt}
                ],
                "temperature": 0.3,  # Lower temperature for more predictable outputs
                "max_tokens": 4000
            }
            
            logger.info(f"Making direct OpenRouter API call for NVIDIA model")
            
            response = requests.post(
                url=settings.llm.openrouter_base_url + "/chat/completions",
                headers=headers,
                json=data
            )
            
            logger.info(f"API call status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"OpenRouter API error: {error_text}")
                raise Exception(f"OpenRouter API error: {response.status_code} - {error_text}")
            
            # Parse the response
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            
            logger.info(f"Got response of length: {len(content)} characters")
            logger.info(f"First 100 characters: {content[:100]}")
            
            # Extract code - look for code blocks
            code = "# No code found in response"
            code_blocks = re.findall(r'```(?:\w+)?\s*([\s\S]*?)```', content)
            if code_blocks:
                code = code_blocks[0].strip()
                logger.info(f"Extracted code block with {len(code)} characters")
            
            # For explanation, just use everything outside code blocks
            # First remove all code blocks
            explanation_text = re.sub(r'```(?:\w+)?[\s\S]*?```', '', content).strip()
            if not explanation_text:
                explanation_text = "No explanation provided. Please try again or use a different model."
            
            # Extract time and space complexity
            time_complexity = "O(n)"  # Default
            space_complexity = "O(n)"  # Default
            
            time_match = re.search(r'[Tt]ime [Cc]omplexity:?\s*O\(([^)]+)\)', content)
            if time_match:
                time_complexity = f"O({time_match.group(1)})"
                
            space_match = re.search(r'[Ss]pace [Cc]omplexity:?\s*O\(([^)]+)\)', content)
            if space_match:
                space_complexity = f"O({space_match.group(1)})"
            
            # Create solution object - using hardcoded values that won't cause display issues
            solution = CodeSolution(
                code=code,
                explanation=explanation_text,
                time_complexity=time_complexity,
                space_complexity=space_complexity,
                edge_cases=["None specified"],
                language=language
            )
            
            logger.info(f"Created CodeSolution with code length: {len(code)}")
            return solution
            
        except Exception as e:
            logger.error(f"Error in NVIDIA direct handler: {e}")
            # Return a minimal valid solution that will display properly
            return CodeSolution(
                code="# Error occurred when calling the NVIDIA model.\n# Please try again or use a different model.\n\n# Error: " + str(e),
                explanation="An error occurred when generating the solution. Please try again or switch to a different model.",
                time_complexity="O(n)",
                space_complexity="O(n)",
                edge_cases=["None specified"],
                language=language
            )
            
    def _clean_json_response(self, content: str) -> str:
        """Clean up common issues in JSON responses"""
        # Remove any markdown code block markers
        content = re.sub(r'```(?:json)?', '', content)
        
        # Remove any text before opening brace and after closing brace
        match = re.search(r'(\{[\s\S]*\})', content)
        if match:
            content = match.group(1)
            
        # Remove trailing commas before closing braces/brackets (common JSON error)
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Fix missing quotes around keys
        content = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', content)
        
        # Trim whitespace
        content = content.strip()
        
        return content
    
    def _attempt_json_repair(self, content: str) -> str:
        """Try to repair common JSON formatting errors"""
        # Check for unbalanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        
        if open_braces > close_braces:
            # Add missing closing braces
            content += '}' * (open_braces - close_braces)
        
        # Check for unbalanced brackets
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        if open_brackets > close_brackets:
            # Add missing closing brackets
            content += ']' * (open_brackets - close_brackets)
            
        # Try to fix missing commas between properties
        content = re.sub(r'"\s*}', '"}', content)  # Remove space between closing quote and brace
        content = re.sub(r'"\s*]', '"]', content)  # Remove space between closing quote and bracket
        
        # Fix newlines in string values
        content = re.sub(r'([^\\])(\n+)', r'\1\\n', content)
        
        return content
        
    def _extract_code_from_text(self, content: str) -> str:
        """Extract code block and other solution components from text content"""
        logger.info("Extracting code from non-JSON response")
        
        # First, remove any <think> tags and their content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Also remove any lone opening think tags (in case closing tag is missing)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        
        # Strategy 1: Look for code between triple backticks (most common format)
        code_blocks = re.findall(r'```(?:\w+)?\s*([\s\S]*?)```', content)
        if code_blocks:
            logger.info(f"Found {len(code_blocks)} code blocks in backticks")
            # Return the longest code block (assuming it's the solution)
            longest_block = max(code_blocks, key=len)
            if longest_block.strip():
                return longest_block.strip()
        
        # Strategy 2: Look for function definitions by language
        code_patterns = [
            # Python function/class definitions
            r'def\s+\w+\s*\([^)]*\)(?:\s*->[\s\w\[\],]*)?:(?:[\s\S]*?)(?:(?=def\s)|$)',
            r'class\s+\w+(?:\s*\([^)]*\))?:(?:[\s\S]*?)(?:(?=class\s)|$)',
            # JavaScript/TypeScript function definitions
            r'(?:function|const|let|var)\s+\w+\s*(?:=\s*(?:function)?\s*)?\([^)]*\)\s*(?:=>)?\s*{(?:[\s\S]*?)}(?:\s*;)?',
            # Java method/class definitions
            r'(?:public|private|protected)(?:\s+static)?\s+(?:\w+(?:<[^>]+>)?)\s+\w+\s*\([^)]*\)\s*{(?:[\s\S]*?)}',
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, content)
            if matches:
                logger.info(f"Found code using language-specific pattern")
                # Join all matches as they might be multiple functions
                return "\n\n".join(match.strip() for match in matches)
        
        # Strategy 3: Analyze lines that likely contain code based on common characteristics
        lines = content.split('\n')
        code_lines = []
        in_code_block = False
        code_indicators = [
            lambda l: re.match(r'^\s*(?:def|class|if|for|while|return|import|from)\s', l),  # Python
            lambda l: re.match(r'^\s*(?:function|const|let|var|if|for|while|return|import|export)\s', l),  # JS
            lambda l: re.match(r'^\s*(?:public|private|protected|class|if|for|while|return|import)\s', l),  # Java
            lambda l: l.strip().endswith('{') or l.strip().endswith('}'),  # Block markers
            lambda l: re.match(r'^\s*[a-zA-Z_]\w*\s*[=:]\s*', l),  # Variable assignments
            lambda l: re.match(r'^\s*[*/]', l),  # Comments
            lambda l: l.strip().startswith('//') or l.strip().startswith('/*') or l.strip().startswith('*'),  # Comments
            lambda l: l.strip().startswith('#'),  # Python comments
        ]
        
        for i, line in enumerate(lines):
            # Check if line likely contains code
            is_code = any(indicator(line) for indicator in code_indicators)
            
            # If indented or previous line was code, likely part of code block
            if i > 0 and (line.startswith(' ' * 4) or line.startswith('\t')):
                is_code = True
            
            # Lines that clearly indicate this is NOT code
            not_code_indicators = [
                'Time Complexity:',
                'Space Complexity:',
                'Explanation:',
                'Edge Cases:',
                'This solution works by',
                'First, we need to',
                'The approach is',
            ]
            
            if any(indicator in line for indicator in not_code_indicators):
                is_code = False
                if in_code_block:
                    # We've reached the end of a code block
                    in_code_block = False
            
            if is_code:
                in_code_block = True
                code_lines.append(line)
            elif in_code_block and line.strip() == '':
                # Keep blank lines within code blocks
                code_lines.append(line)
            else:
                in_code_block = False
        
        if code_lines:
            logger.info("Extracted code using line-by-line heuristic analysis")
            return '\n'.join(code_lines)
            
        # Strategy 4: Extract everything before "Time Complexity" or similar markers
        # This is a last resort when we can't identify clear code patterns
        code_end_markers = [
            'Time Complexity:',
            'Space Complexity:',
            'Explanation:',
            'Edge Cases:',
            'Analysis:',
        ]
        
        for marker in code_end_markers:
            if marker in content:
                code_section = content.split(marker)[0].strip()
                if code_section:
                    logger.info(f"Extracted code by splitting at '{marker}'")
                    return code_section
                
        # If all strategies fail, log warning and return original content or error message
        logger.warning("Could not extract code using any method")
        return "# No code could be extracted from the model's response\n# Please try rephrasing your question or using a different model"

    def _extract_explanation_from_text(self, content: str) -> str:
        """Extract explanation section from text response"""
        logger.info("Trying to extract explanation from response")
        
        # Try to match explanation section with different formats
        explanation_patterns = [
            # Format: "Explanation: text here"
            r'(?:explanation|approach|solution explanation)[:]\s*([\s\S]*?)(?:(?:time|space) complexity|edge cases|\Z)',
            # Format: "Time complexity... Space complexity... Here is the explanation..." 
            r'(?:time\s+complexity|space\s+complexity).*?(?:(?:time|space) complexity).*?(?:here is|this solution|this approach|the explanation)(?:\s+is)?[:\s]+([\s\S]*?)(?:edge cases|\Z)',
            # Any text after complexity sections that isn't edge cases
            r'(?:time\s+complexity|space\s+complexity).*?(?:(?:time|space) complexity).*?([\s\S]*?)(?:edge cases|\Z)'
        ]
        
        # Try each pattern
        for pattern in explanation_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match and match.group(1).strip():
                explanation = match.group(1).strip()
                logger.info(f"Found explanation using pattern: {pattern[:30]}...")
                return explanation
                
        # If we reach here, try to extract anything between the complexity analysis and edge cases
        time_space_match = re.search(r'(?:time\s+complexity|space\s+complexity).*?(?:(?:time|space) complexity)', content, re.IGNORECASE | re.DOTALL)
        edge_cases_match = re.search(r'edge\s+cases', content, re.IGNORECASE)
        
        if time_space_match and edge_cases_match:
            # Extract content between complexity sections and edge cases
            start_pos = time_space_match.end()
            end_pos = edge_cases_match.start()
            if start_pos < end_pos:
                explanation = content[start_pos:end_pos].strip()
                if explanation:
                    logger.info("Extracted explanation as text between complexity and edge cases")
                    return explanation
                    
        # Look for any paragraph after the code block
        code_block_end = content.find("```\n")
        if code_block_end != -1:
            # Get the text after the code block but before time/space complexity
            complexity_match = re.search(r'(?:time|space)\s+complexity', content[code_block_end:], re.IGNORECASE)
            if complexity_match:
                start_pos = code_block_end + 4  # Skip past the closing backticks
                end_pos = code_block_end + complexity_match.start()
                potential_explanation = content[start_pos:end_pos].strip()
                if potential_explanation and len(potential_explanation) > 30:  # Must be substantial
                    logger.info("Extracted explanation from text between code and complexity")
                    return potential_explanation
        
        # Last resort: Just find any substantial paragraph that might be an explanation
        paragraphs = re.split(r'\n\s*\n', content)
        for paragraph in paragraphs:
            cleaned = paragraph.strip()
            # Skip paragraphs that are likely code or section headers
            if (cleaned and len(cleaned) > 50 and 
                not cleaned.startswith('```') and 
                not re.match(r'^(time|space) complexity|^edge cases', cleaned, re.IGNORECASE)):
                logger.info("Used fallback paragraph extraction for explanation")
                return cleaned
                
        logger.warning("Could not extract any explanation")
        return "No explanation was provided in the model's response."
