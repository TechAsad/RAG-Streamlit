from flask import Flask, request, jsonify
from rest_api_files.main_api_call import RAGbot  # Update 'your_module_name' with the actual module name containing RAGbot

app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def ask_question():
    data = request.json
    prompt = data['prompt']
    memory = data['memory'] #need to pass memory variable to chatbot. if not, pass empty memory.
    response = RAGbot.run(prompt, memory)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
