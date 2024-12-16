
import os
import base64

def get_api_keys():
    """Get API keys from environment variables with verification"""
    try:
        keys = {
            'openai': os.environ.get('OPENAI_KEY', ''),
            'anthropic': os.environ.get('ANTHROPIC_KEY', ''),
            'qdrant': os.environ.get('QDRANT_KEY', '')
        }
        
        # Verify keys match expected format
        assert keys['openai'].startswith('sk-proj-'), "OpenAI key format invalid"
        assert keys['anthropic'].startswith('sk-ant-'), "Anthropic key format invalid"
        assert len(keys['qdrant']) > 0, "Qdrant key invalid"
        
        return keys
    except Exception as e:
        raise ValueError(f"Error processing API keys: {str(e)}")

if __name__ == "__main__":
    try:
        keys = get_api_keys()
        print("API keys verified successfully")
    except ValueError as e:
        print(str(e))
