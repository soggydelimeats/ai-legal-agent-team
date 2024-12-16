
import os

def get_api_keys():
    """Get API keys from environment variables"""
    keys = {
        'openai': os.environ.get('OPENAI_KEY'),
        'anthropic': os.environ.get('ANTHROPIC_KEY'),
        'qdrant': os.environ.get('QDRANT_KEY')
    }
    
    # Check if we're in deployment environment
    if os.environ.get('REPLIT_DEPLOYMENT') and not any(keys.values()):
        raise ValueError("API keys not found in deployment environment")
        
    return keys

if __name__ == "__main__":
    keys = get_api_keys()
    print("API keys loaded")
