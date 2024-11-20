from flask import Flask, request, jsonify
import logging
import time
import os
from collections import defaultdict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
import requests
api_key = os.getenv('OPENAI_API_KEY')

print(f"API key loaded (last 4 chars): ...{api_key[-4:]}")

client = OpenAI(api_key=api_key)

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

message_buffer = defaultdict(list)
last_print_time = defaultdict(float)
AGGREGATION_INTERVAL = 10  
notification_cooldowns = defaultdict(float)
NOTIFICATION_COOLDOWN = 30  
CRYPTO_API_KEY = os.getenv('CRYPTO_API_KEY')
CRYPTO_API_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'

def get_crypto_price(slug: str, currency: str = 'USD') -> dict:
    """
    Get the current price of a cryptocurrency.
    
    Args:
        slug (str): The slug/name of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
        currency (str, optional): The currency to get the price in. Defaults to 'USD'
    
    Returns:
        dict: A dictionary containing the price data and status
            {
                'success': bool,
                'price': float or None,
                'error': str or None,
                'currency': str
            }
    """
    headers = {
        'X-CMC_PRO_API_KEY': CRYPTO_API_KEY,
        'Accept': 'application/json'
    }
    params = {
        'slug': slug.lower(),
        'convert': currency.upper()
    }
    
    try:
        response = requests.get(CRYPTO_API_URL, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        print(data)
        if data.get('status', {}).get('error_code') != 0:
            return {
                'success': False,
                'price': None,
                'error': data.get('status', {}).get('error_message'),
                'currency': currency.upper()
            }
        price = data['data'][list(data['data'].keys())[-1]]['quote'][currency.upper()]['price']
        return {
            'success': True,
            'price': price,
            'error': None,
            'currency': currency.upper()
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'price': None,
            'error': f"API request failed: {str(e)}",
            'currency': currency.upper()
        }
    except KeyError as e:
        return {
            'success': False,
            'price': None,
            'error': "Cryptocurrency or currency not found",
            'currency': currency.upper()
        }
    except Exception as e:
        return {
            'success': False,
            'price': None,
            'error': f"Unexpected error: {str(e)}",
            'currency': currency.upper()
        }

# Add the function definition for OpenAI
crypto_price_function = {
    "name": "get_crypto_price",
    "description": "Get the current price of a cryptocurrency in a specified currency",
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "The slug/name of the cryptocurrency (e.g., 'bitcoin', 'ethereum')"
            },
            "currency": {
                "type": "string",
                "description": "The currency to get the price in (e.g., 'USD', 'EUR')",
                "default": "USD"
            }
        },
        "required": ["slug"]
    }
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyse_crypto_price(text):
    """Analyze text for cryptocurrency price requests using OpenAI"""
    try:
        logger.info("Attempting to connect to OpenAI API...")
        if not api_key:
            raise ValueError("OpenAI API key is not set")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI that helps users get cryptocurrency prices. \
                    Analyze the user's message and if they're asking for a crypto price, \
                        use the get_crypto_price function to fetch it."},
                {"role": "user", "content": text}
            ],
            functions=[crypto_price_function],
            function_call="auto",
            temperature=0.7,
            max_tokens=150
        )
        
        message = response.choices[0].message
        print(message)
        # Check if the model wants to call the function
        if message.function_call:
            # Parse the function arguments
            import json
            function_args = json.loads(message.function_call.arguments)
            
            # Call the get_crypto_price function
            result = get_crypto_price(
                slug=function_args.get('slug'),
                currency=function_args.get('currency', 'USD')
            )
            
            if result['success']:
                return {
                    'is_crypto_request': True,
                    'price': result['price'],
                    'currency': result['currency'],
                    'crypto': function_args.get('slug'),
                    'error': None,
                    'message': "The price of " + function_args.get('slug') + " in " + function_args.get('currency','USD') + ": " + str(result['price'].__round__(2))
                }
            else:
                return {
                    'is_crypto_request': False,
                    'price': None,
                    'currency': result['currency'],
                    'crypto': function_args.get('slug'),
                    'error': result['error'],
                    'message': "I couldn't find the price of " + function_args.get('slug') + " in " + function_args.get('currency','USD') + ". Please try again with a different cryptocurrency or currency."
                }
        
        return {
            'is_crypto_request': False,
            'price': None,
            'currency': None,
            'crypto': None,
            'error': None,
            'message': "I couldn't find the price of that cryptocurrency. Please try again with a different cryptocurrency or currency."
        }

    except Exception as e:
        logger.error(f"Error analyzing crypto price request: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            'is_crypto_request': False,
            'price': None,
            'currency': None,
            'crypto': None,
            'error': str(e)
        }

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        logger.info("Received webhook POST request")
        data = request.json
        logger.info(f"Received data: {data}")
        
        session_id = data.get('session_id')
        if not session_id:
            logger.error("No session_id provided in request")
            return jsonify({"status": "error", "message": "No session_id provided"}), 400
            
        segments = data.get('segments', [])
        logger.info(f"Processing session_id: {session_id}, number of segments: {len(segments)}")
        
        current_time = time.time()
        
        time_since_last_notification = current_time - notification_cooldowns[session_id]
        if time_since_last_notification < NOTIFICATION_COOLDOWN:
            logger.info(f"Notification cooldown active for session {session_id}. {NOTIFICATION_COOLDOWN - time_since_last_notification:.0f}s remaining")
            return jsonify({"status": "success"}), 200
        
        for segment in segments:
            if segment['text']: 
                message_buffer[session_id].append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'speaker': segment['speaker']
                })
                logger.info(f"Added segment text for session {session_id}: {segment['text']}")
        
        time_since_last = current_time - last_print_time[session_id]
        logger.info(f"Time since last process: {time_since_last}s (threshold: {AGGREGATION_INTERVAL}s)")
        
        if time_since_last >= AGGREGATION_INTERVAL and message_buffer[session_id]:
            logger.info(f"Processing aggregated messages for session {session_id}...")
            sorted_messages = sorted(message_buffer[session_id], key=lambda x: x['start'])
            combined_text = ' '.join(msg['text'] for msg in sorted_messages if msg['text'])
            logger.info(f"Analyzing combined text for session {session_id}: {combined_text}")
            
            message_buffer[session_id].clear()
            last_print_time[session_id] = current_time
            analysis = analyse_crypto_price(combined_text)
            if analysis["is_crypto_request"]:
                logger.warning(f"🚨 Crypto price request detected for session {session_id}!")
                notification_cooldowns[session_id] = current_time
                return jsonify({
                    "message": analysis["message"]
                }), 200
        
        return jsonify({"status": "success"}), 200

@app.route('/webhook/setup-status', methods=['GET'])
def setup_status():
    try:
        return jsonify({
            "is_setup_completed": True
        }), 200
    except Exception as e:
        logger.error(f"Error checking setup status: {str(e)}")
        return jsonify({
            "is_setup_completed": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)