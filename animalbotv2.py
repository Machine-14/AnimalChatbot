from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss  # Import FAISS for efficient similarity search
import json
import requests

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Neo4j connection details
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

# Store chat history in Neo4j
def store_chat_history(user_id, question_text, response_text):
    query = '''
    MERGE (u:User {user_id: $user_id})
    CREATE (q:Question {text: $question_text})
    CREATE (r:Response {text: $response_text})
    MERGE (u)-[:ASKED]->(q)
    MERGE (q)-[:REPLIED_WITH]->(r)
    '''
    parameters = {'user_id': user_id, 'question_text': question_text, 'response_text': response_text}
    run_query(query, parameters)

# Cypher query to retrieve greetings from the Neo4j database
cypher_query = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''

# Retrieve greetings from the Neo4j database
greeting_corpus = []
results = run_query(cypher_query)
for record in results:
    greeting_corpus.append(record['name'])

# Ensure corpus is unique
greeting_corpus = list(set(greeting_corpus))

# Encode the greeting corpus into vectors using the sentence transformer model
greeting_vecs = model.encode(greeting_corpus, convert_to_numpy=True, normalize_embeddings=True)

# Initialize FAISS index
d = greeting_vecs.shape[1]  # Dimension of vectors
index = faiss.IndexFlatL2(d)  # L2 distance index (cosine similarity can be used with normalization)
index.add(greeting_vecs)  # Add vectors to FAISS index

def compute_similar_faiss(sentence):
    # Encode the query sentence
    ask_vec = model.encode([sentence], convert_to_numpy=True, normalize_embeddings=True)
    # Search FAISS index for nearest neighbor
    D, I = index.search(ask_vec, 1)  # Return top 1 result
    return D[0][0], I[0][0]

def neo4j_search(neo_query):
    results = run_query(neo_query)
    if results:
        response_msg = results[0]['reply']
    else:
        response_msg = "ไม่พบคำตอบในฐานข้อมูล"
    return response_msg

# Ollama API endpoint (assuming you're running Ollama locally)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json"
}

def llama_generate_response(prompt):
    # ปรับ prompt เพื่อให้ได้คำตอบสั้น ๆ
    prompt = f"คำถาม: {prompt}\nตอบสั้น ๆ และได้ใจความ:"
    
    # เตรียม payload สำหรับ API เรียก Ollama
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",  # ปรับชื่อโมเดลได้ตามต้องการ
        "prompt": prompt+"ขอเป็นคำตอบภาษาไทยไม่เกิน20คำ ตอบโดยผู้เชี่ยวชาญด้านกฏหมายสัตว์",
        "stream": False
    }

    # ส่ง POST request ไปยัง Ollama API
    response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))

    # เช็คว่าคำขอสำเร็จหรือไม่
    if response.status_code == 200:
        response_data = response.text
        data = json.loads(response_data)
        decoded_text = data.get("response", "ไม่พบคำตอบ")
        return decoded_text
    else:
        print(f"Failed to get a response: {response.status_code}, {response.text}")
        return "เกิดข้อผิดพลาดในการสร้างคำตอบ"

def compute_response(sentence, user_id):
    score, idx = compute_similar_faiss(sentence)
    if score > 0.5:
        # ใช้วิธี API ของ Ollama สร้างคำตอบที่สั้นและได้ใจความ
        prompt = f"{sentence}"
        my_msg = llama_generate_response(prompt)
        source = " (ตอบโดย Ollama)"
        my_msg_with_source = my_msg + source
    else:
        # ค้นหาข้อความที่ตรงกับ greeting ใน Neo4j
        Match_greeting = greeting_corpus[idx]
        My_cypher = f"MATCH (n:Greeting) WHERE n.name = '{Match_greeting}' RETURN n.msg_reply as reply"
        my_msg = neo4j_search(My_cypher)
        my_msg_with_source = my_msg

    # จัดเก็บประวัติการสนทนาใน Neo4j (คำถามจากผู้ใช้และคำตอบจากบอท)
    store_chat_history(user_id, sentence, my_msg_with_source)

    print(my_msg_with_source)
    return my_msg_with_source

app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = 'access_token'
        secret = 'secret'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        msg = json_data['events'][0]['message']['text']
        user_id = json_data['events'][0]['source']['userId']  # Get user ID
        tk = json_data['events'][0]['replyToken']
        response_msg = compute_response(msg, user_id)
        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
        print(msg, tk)
    except Exception as e:
        print(body)
        print(f"Error: {e}")
    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)
