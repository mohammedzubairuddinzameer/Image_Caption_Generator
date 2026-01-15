import os
from dotenv import load_dotenv

# Force reload .env
load_dotenv(override=True)

token = os.getenv('HF_TOKEN')

print('=' * 60)
print('TOKEN DIAGNOSTIC')
print('=' * 60)
print(f'Token found: {bool(token)}')
print(f'Token length: {len(token) if token else 0}')
print(f'Token starts with hf_: {token.startswith("hf_") if token else False}')
print(f'Token preview: {token[:15] if token else "NONE"}...')
print(f'Token has spaces: {" " in token if token else "N/A"}')
print('=' * 60)

# Test with API
if token:
    import requests
    print('\nTesting with Hugging Face API...')
    
    # Test 1: Check token validity
    response = requests.get(
        'https://huggingface.co/api/whoami-v2',
        headers={'Authorization': f'Bearer {token}'}
    )
    
    print(f'Auth test status: {response.status_code}')
    if response.status_code == 200:
        print('TOKEN IS VALID')
        data = response.json()
        print(f'Username: {data.get("name")}')
    else:
        print('TOKEN IS INVALID')
        print(f'Response: {response.text[:200]}')
    
    # Test 2: Try inference API
    print('\nTesting Inference API...')
    test_response = requests.post(
        'https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base',
        headers={'Authorization': f'Bearer {token}'},
        json={'inputs': 'test'}
    )
    
    print(f'Inference API status: {test_response.status_code}')
    if test_response.status_code == 401:
        print('401 Error - Check token permissions!')
    elif test_response.status_code == 503:
        print('Model loading - wait 20 seconds')
    elif test_response.status_code == 400:
        print('API accessible (400 is ok for bad data)')
    else:
        print(f'Response: {test_response.text[:200]}')
else:
    print('NO TOKEN FOUND IN .env FILE!')

print('=' * 60)
