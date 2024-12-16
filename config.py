
import os

def get_api_keys():
    """Get API keys from environment variables"""
    return {
        'openai': os.environ.get('OPENAI_KEY', ''),
        'anthropic': os.environ.get('ANTHROPIC_KEY', ''),
        'qdrant': os.environ.get('QDRANT_KEY', '')
    }

if __name__ == "__main__":
    keys = get_api_keys()
    print("API keys loaded")
