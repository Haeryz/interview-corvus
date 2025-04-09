from typing import List, Type, TypeVar

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
        """Special dedicated handler for NVIDIA model that bypasses LlamaIndex entirely"""
        try:
            # Get the OpenRouter API key
            api_key = APIKeyManager().get_api_key()
            
            # Prepare the prompt
            prompt_manager = PromptManager()
            system_prompt = "detailed thinking on"
            user_prompt = prompt_manager.get_prompt(
                "code_solution", problem_description=problem_description, language=language
            )
            
            logger.info(f"Using NVIDIA model with system prompt: {system_prompt}")
            
            import requests
            import json
            
            # Set up headers for OpenRouter API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": settings.llm.openrouter_site_url,
                "X-Title": settings.llm.openrouter_site_name
            }
            
            # Structure the API request with NVIDIA-specific settings
            # Based directly on NVIDIA's recommended parameters
            data = {
                "model": settings.llm.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.6,  # NVIDIA recommended for reasoning mode
                "top_p": 0.95,       # NVIDIA recommended for reasoning mode
                "max_tokens": 4000,
                "response_format": {"type": "json_object"}
            }
            
            logger.info(f"Making direct OpenRouter API call for NVIDIA Llama model")
            
            # Make the API call
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
            
            try:
                # Try to parse as JSON
                logger.info("Parsing JSON response")
                solution_data = json.loads(content)
                
                # Get edge cases - either as list or convert string to list
                edge_cases = solution_data.get("edge_cases", [])
                if isinstance(edge_cases, str):
                    if edge_cases.lower() in ["none", "n/a", "none specified"]:
                        edge_cases = ["None specified"]
                    else:
                        edge_cases = [edge_cases]
                elif not isinstance(edge_cases, list):
                    edge_cases = ["None specified"]
                    
                # Create a CodeSolution object
                solution = CodeSolution(
                    code=solution_data.get("code", "# No code provided in response"),
                    explanation=solution_data.get("explanation", "No explanation provided"),
                    time_complexity=solution_data.get("time_complexity", "O(n)"),
                    space_complexity=solution_data.get("space_complexity", "O(n)"),
                    edge_cases=edge_cases,
                    language=language
                )
                return solution
            except json.JSONDecodeError:
                # If not valid JSON, create a simple solution with the raw text
                logger.warning("Response was not valid JSON, using raw text content")
                logger.info(f"Non-JSON response: {content[:100]}...")
                solution = CodeSolution(
                    code=f"# Response format error - raw response:\n\n{content[:1000]}...",
                    explanation="The model did not return proper JSON. See raw response in code section.",
                    time_complexity="Unknown",
                    space_complexity="Unknown",
                    edge_cases=["Unknown"],
                    language=language
                )
                return solution
        except Exception as direct_api_error:
            logger.error(f"Error using direct API for NVIDIA: {direct_api_error}")
            return CodeSolution(
                code=f"# Error with NVIDIA model\n# {str(direct_api_error)}\n\n# Try using a different model",
                explanation=f"Error occurred with NVIDIA model: {str(direct_api_error)}",
                time_complexity="N/A",
                space_complexity="N/A",
                edge_cases=["N/A"],
                language=language
            )
