import requests

base_url = 'http://localhost:8080'

def get_server_health():
    response = requests.get(f'{base_url}/health')
    return response.json()

def post_completion(context, user_input):
    prompt = f"{context}\nUser: {user_input}\nAssistant:"
    data = {
        'prompt': prompt,
        #creativity
        'temperature': 0.8,
        #randomness
        'top_k': 35,
        #diversity
        'top_p': 0.95,
        #limits the amount of words generated
        'n_predict': 400,
        #stop when seeing specific words
        'stop': ["</s>", "Assistant:", "User:"]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f'{base_url}/completion', json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['content'].strip()
    else:
        return "Error processing your request. Please try again."
    
def update_context(context, user_input, assistant_response):
    return f"{context}\nUser: {user_input}\nAssistant: {assistant_response}"

def main():
    context = "You are a friendly AI assistant designed to provide helpful, succinct, and accurate information."

    health = get_server_health()
    print('Server Health:', health)

    if health.get('status') == 'ok':
        while True:
            user_input = input("Enter a prompt or type 'exit' to quit: ")
            if user_input.lower() == 'exit':
                break
            assistant_response = post_completion(context, user_input)
            print('Assistant:', assistant_response)

            context = update_context(context, user_input, assistant_response)
    else:
        print("Server is not ready for requests.")

if __name__ == "__main__":
    main()