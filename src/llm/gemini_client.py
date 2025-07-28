"""
Gemini API Client for KBLI Classification

This module provides a client for interacting with Google's Gemini API
with built-in rate limiting, retry logic, and multi-model support.
"""

import os
import time
import json
import logging
from typing import Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    """
    
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.
        
        Parameters
        ----------
        requests_per_minute : int
            Maximum requests allowed per minute.
        """
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_refill = time.time()
        self.request_times = deque()
        
        logger.debug(f"Rate limiter initialized: {requests_per_minute} requests/minute")
    
    def can_make_request(self) -> bool:
        """
        Check if a request can be made without hitting rate limits.
        
        Returns
        -------
        bool
            True if request can be made, False otherwise.
        """
        now = time.time()
        
        # Remove requests older than 1 minute
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Check if we can make a request
        return len(self.request_times) < self.requests_per_minute
    
    def wait_time(self) -> float:
        """
        Calculate how long to wait before next request.
        
        Returns
        -------
        float
            Wait time in seconds.
        """
        if self.can_make_request():
            return 0.0
        
        # Wait until the oldest request is 1 minute old
        if self.request_times:
            oldest_request = self.request_times[0]
            wait_time = 60 - (time.time() - oldest_request)
            return max(0, wait_time)
        
        return 0.0
    
    def record_request(self):
        """Record that a request was made."""
        self.request_times.append(time.time())
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        wait_time = self.wait_time()
        if wait_time > 0:
            logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)


class GeminiClient:
    """
    Google Gemini API client with rate limiting and retry logic.
    """
    
    # Model configurations with their rate limits (requests per minute)
    MODEL_CONFIGS = {
        'gemini-2.0-flash-exp': {
            'requests_per_minute': 15,
            'display_name': 'Gemini 2.0 Flash Experimental'
        },
        'gemini-1.5-flash': {
            'requests_per_minute': 15,
            'display_name': 'Gemini 1.5 Flash'
        },
        'gemini-1.5-pro': {
            'requests_per_minute': 2,
            'display_name': 'Gemini 1.5 Pro'
        },
        'gemini-1.5-flash-8b': {
            'requests_per_minute': 15,
            'display_name': 'Gemini 1.5 Flash 8B'
        }
    }
    
    def __init__(
        self,
        model_name: str = 'gemini-2.0-flash-exp',
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Gemini client.
        
        Parameters
        ----------
        model_name : str, default 'gemini-2.0-flash-exp'
            Name of the Gemini model to use.
        temperature : float, default 0.7
            Sampling temperature for text generation.
        api_key : str, optional
            Gemini API key. If None, loads from environment or .env file.
        max_retries : int, default 3
            Maximum number of retry attempts for failed requests.
        retry_delay : float, default 1.0
            Base delay between retries (exponential backoff).
        
        Raises
        ------
        ValueError
            If model is not supported or API key is missing.
        """
        
        # Validate model
        if model_name not in self.MODEL_CONFIGS:
            available_models = ', '.join(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Unsupported model '{model_name}'. Available models: {available_models}")
        
        self.model_name = model_name
        self.model_config = self.MODEL_CONFIGS[model_name]
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Load API key
        self.api_key = self._load_api_key(api_key)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(self.model_config['requests_per_minute'])
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                candidate_count=1,
                max_output_tokens=2048,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        logger.info(f"Gemini client initialized: {self.model_config['display_name']} "
                   f"({self.model_config['requests_per_minute']} req/min, temp={temperature})")
    
    def _load_api_key(self, api_key: Optional[str] = None) -> str:
        """
        Load API key from parameter, environment, or .env file.
        
        Parameters
        ----------
        api_key : str, optional
            API key to use. If None, loads from environment.
        
        Returns
        -------
        str
            Loaded API key.
            
        Raises
        ------
        ValueError
            If API key cannot be found.
        """
        
        if api_key:
            return api_key
        
        # Try environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            logger.debug("API key loaded from environment variable")
            return api_key
        
        # Try .env file
        env_path = Path.cwd() / '.env'
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    logger.debug("API key loaded from .env file")
                    return api_key
            except ImportError:
                logger.warning("python-dotenv not available, skipping .env file")
        
        raise ValueError(
            "Gemini API key not found. Please provide it via:\n"
            "1. Parameter: GeminiClient(api_key='your_key')\n"
            "2. Environment variable: GEMINI_API_KEY\n"
            "3. .env file with GEMINI_API_KEY=your_key"
        )
    
    def generate_response(self, prompt: str) -> Dict:
        """
        Generate a response from Gemini with retry logic and rate limiting.
        
        Parameters
        ----------
        prompt : str
            Input prompt for the model.
        
        Returns
        -------
        Dict
            Response containing 'success', 'content', 'error', 'metadata' fields.
            
        Examples
        --------
        >>> client = GeminiClient()
        >>> response = client.generate_response("Classify this job description...")
        >>> if response['success']:
        ...     print(response['content'])
        """
        
        for attempt in range(self.max_retries + 1):
            try:
                # Wait for rate limit if needed
                self.rate_limiter.wait_if_needed()
                
                # Record the request
                self.rate_limiter.record_request()
                
                logger.debug(f"Making API request (attempt {attempt + 1}/{self.max_retries + 1})")
                
                # Make the API call
                start_time = time.time()
                response = self.model.generate_content(prompt)
                end_time = time.time()
                
                # Check if response was blocked
                if not response.text:
                    if response.candidates and response.candidates[0].finish_reason:
                        finish_reason = response.candidates[0].finish_reason.name
                        logger.warning(f"Response blocked: {finish_reason}")
                        return {
                            'success': False,
                            'content': None,
                            'error': f'Response blocked: {finish_reason}',
                            'metadata': {
                                'model': self.model_name,
                                'temperature': self.temperature,
                                'attempt': attempt + 1,
                                'response_time': end_time - start_time,
                                'timestamp': datetime.now().isoformat()
                            }
                        }
                
                # Successful response
                logger.debug(f"API request successful (response time: {end_time - start_time:.2f}s)")
                
                return {
                    'success': True,
                    'content': response.text,
                    'error': None,
                    'metadata': {
                        'model': self.model_name,
                        'temperature': self.temperature,
                        'attempt': attempt + 1,
                        'response_time': end_time - start_time,
                        'timestamp': datetime.now().isoformat(),
                        'usage': {
                            'prompt_tokens': response.usage_metadata.prompt_token_count if response.usage_metadata else None,
                            'completion_tokens': response.usage_metadata.candidates_token_count if response.usage_metadata else None,
                            'total_tokens': response.usage_metadata.total_token_count if response.usage_metadata else None
                        }
                    }
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API request failed (attempt {attempt + 1}): {error_msg}")
                
                # Check for quota exhaustion
                if 'quota' in error_msg.lower() or 'limit' in error_msg.lower():
                    logger.error("API quota exhausted or rate limit exceeded")
                    return {
                        'success': False,
                        'content': None,
                        'error': f'Quota exhausted: {error_msg}',
                        'metadata': {
                            'model': self.model_name,
                            'temperature': self.temperature,
                            'attempt': attempt + 1,
                            'timestamp': datetime.now().isoformat(),
                            'quota_exhausted': True
                        }
                    }
                
                # If this is the last attempt, return the error
                if attempt == self.max_retries:
                    logger.error(f"All retry attempts failed: {error_msg}")
                    return {
                        'success': False,
                        'content': None,
                        'error': error_msg,
                        'metadata': {
                            'model': self.model_name,
                            'temperature': self.temperature,
                            'attempt': attempt + 1,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                
                # Wait before retry with exponential backoff
                wait_time = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model configuration.
        
        Returns
        -------
        Dict
            Model configuration information.
        """
        
        return {
            'model_name': self.model_name,
            'display_name': self.model_config['display_name'],
            'requests_per_minute': self.model_config['requests_per_minute'],
            'temperature': self.temperature,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay
        }
    
    @classmethod
    def get_available_models(cls) -> Dict:
        """
        Get list of available models and their configurations.
        
        Returns
        -------
        Dict
            Available models and their configurations.
        """
        
        return cls.MODEL_CONFIGS.copy()


# Convenience functions
def create_gemini_client(
    model_name: str = 'gemini-2.0-flash-exp',
    temperature: float = 0.7,
    **kwargs
) -> GeminiClient:
    """
    Create a GeminiClient instance with default settings.
    
    Parameters
    ----------
    model_name : str, default 'gemini-2.0-flash-exp'
        Name of the Gemini model to use.
    temperature : float, default 0.7
        Sampling temperature for text generation.
    **kwargs
        Additional arguments passed to GeminiClient.
    
    Returns
    -------
    GeminiClient
        Initialized Gemini client.
    """
    
    return GeminiClient(model_name=model_name, temperature=temperature, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        print("Testing Gemini Client...")
        
        # Show available models
        models = GeminiClient.get_available_models()
        print(f"\nAvailable models:")
        for model, config in models.items():
            print(f"  {model}: {config['display_name']} ({config['requests_per_minute']} req/min)")
        
        # Create client
        client = GeminiClient(model_name='gemini-2.0-flash-exp', temperature=0.7)
        
        # Get model info
        info = client.get_model_info()
        print(f"\nClient info: {info}")
        
        # Test simple request
        test_prompt = "Hello! Please respond with a simple JSON object containing 'message': 'Hello World'"
        
        print(f"\nTesting API request...")
        response = client.generate_response(test_prompt)
        
        if response['success']:
            print(f"✅ API request successful!")
            print(f"Response: {response['content'][:100]}...")
            print(f"Metadata: {response['metadata']}")
        else:
            print(f"❌ API request failed: {response['error']}")
            if response['metadata'].get('quota_exhausted'):
                print("⚠️  Quota exhausted - this is expected behavior")
        
        print("\n✅ Gemini client testing completed!")
        
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        print(f"❌ Testing failed: {str(e)}")
