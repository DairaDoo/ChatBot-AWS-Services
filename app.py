from flask import Flask, render_template, request, jsonify
import os
import sys
from pathlib import Path

# Agregar el directorio actual al path para importar módulos locales
sys.path.append(os.getcwd())

# Importar el sistema RAG
from rag_system import RAGSystem

app = Flask(__name__)

# Inicializar el sistema RAG con la ruta absoluta del archivo de conocimiento
script_dir = Path(__file__).parent
knowledge_path = script_dir / "data" / "knowledge_base.txt"
rag_system = RAGSystem(str(knowledge_path))

@app.route('/')
def index():
    """Renderiza la página principal del chatbot."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint para procesar las consultas del usuario."""
    # Obtener datos de la solicitud
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message.strip():
        return jsonify({'response': 'Por favor, escribe un mensaje.'})
    
    # Generar respuesta usando el sistema RAG
    response = rag_system.generate_response(user_message)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)