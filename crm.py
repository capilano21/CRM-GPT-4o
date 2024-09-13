import streamlit as st
import pickle
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import assemblyai as aai
import asyncio
import websockets
import base64
import json
import pyaudio

# Set your API keys
os.environ["OPENAI_API_KEY"] = # Your KEY HERE #
aai.settings.api_key = #Your KEY HERE #
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
auth_key = "e6707389a0584af48bdc0fc3e387b74a"
# Function to load data from pickle files
@st.cache_data
def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load the dictionaries and cache them
summary_dict = load_data_from_pickle('summary_dict.pkl')
data_dict = load_data_from_pickle('data_dict.pkl')

keys = [
    'employee_records', 'customer_orders_pending', 'product_orders_pending',
    'circle_orders_pending', 'product_inventory', 'customer_visit',
    'employee_performance', 'customer_records', 'Freestyle'
]

# Polished names for the buttons
button_names = {
    'employee_records': 'Employee Records',
    'customer_orders_pending': 'Customer Orders Pending',
    'product_orders_pending': 'Product Orders Pending',
    'circle_orders_pending': 'Circle Orders Pending',
    'product_inventory': 'Product Inventory',
    'customer_visit': 'Customer Visit',
    'employee_performance': 'Employee Performance',
    'customer_records': 'Customer Records',
    'Freestyle': 'Freestyle Mode'
}

# Define the breakdown function
def breakdown(query, summary_dict=summary_dict):
    class RouteQuery(BaseModel):
        query: str = Field(..., description="The user's query.")
        tables: Dict[str, str] = Field(description="A dictionary where keys are table names and values are table descriptions.")

        def route(self, llm: ChatOpenAI) -> List[str]:
            combined_prompt = f"Query: {self.query}\n\n"
            for table_name, description in self.tables.items():
                combined_prompt += f"Table: {table_name}\nDescription: {description}\n\n"
            combined_prompt += "Break down the query into sub-questions that can be found in the provided tables. Return only the concise sub-questions strictly in a Python list. No additional text or explanation."
            response = llm(combined_prompt)
            relevant_tables = response.content
            return relevant_tables

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at breaking questions into parts"),
        ("human", "{input}"),
    ])
    
    class RouteQueryResponse(BaseModel):
        relevant_tables: List[str] = Field(..., description="A list of tables from the dict keys.")

    structured_llm = llm.with_structured_output(RouteQueryResponse)
    router = prompt_template | structured_llm

    datasource = router.invoke({
        "input": f" The following user query {query} may have answers in multiple tables.Here are the tables and descriptions {summary_dict}. Route the appropriate tables."
    })

    return datasource.relevant_tables, None

# Define the function to process the question
def process_question(question, key, data_dict=data_dict):
    if key == "Freestyle":
        table, _ = breakdown(question)
        response = ""
        for entry in table:
            data = data_dict.get(entry, "")
            prompt = f"Use {data} to answer the following question, breakdown the question and only answer the parts that are present in the context. {question}"
            ans = llm.invoke(prompt).content
            response += "\n" + ans
        prompt = f"Use the following data {response} to answer {question}. Note that sometimes the answer will be present below or above even if the text says no info provided."
        answer = llm.invoke(prompt).content
    else:
        data = data_dict.get(key, "")
        prompt = f"Use {data} to answer the following question {question}"
        answer = llm.invoke(prompt).content
    return answer

# Function to record audio using sounddevice
def record_audio():
    fs = 44100  # Sample rate
    duration = 5  # Duration in seconds

    if 'recording_path' not in st.session_state:
        st.session_state.recording_path = None
    if 'audio_transcript' not in st.session_state:
        st.session_state.audio_transcript = ""

    if st.button("Start Recording"):
        st.write("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
        sd.wait()  # Wait until recording is finished
        st.session_state.recording_path = "audio.wav"
        wav.write("audio.wav", fs, recording)
    
    if st.button("Stop Recording"):
        st.write("Recording stopped.")
        # Transcribe the audio
        if st.session_state.recording_path:
            st.session_state.audio_transcript = transcribe_audio(st.session_state.recording_path)
            st.write("### Transcribed Question:")
            st.write(st.session_state.audio_transcript)

    return st.session_state.audio_transcript

# Function to transcribe the audio
def transcribe_audio(audio_file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file_path)
    return transcript.text

# Async function to handle real-time transcription
URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

async def send_receive(stream=stream):
    print(f'Connecting websocket to url {URL}')
    async with websockets.connect(
        URL,
        extra_headers=(("Authorization", auth_key),),
        ping_interval=5,
        ping_timeout=20
    ) as _ws:
        # Initial delay
        await asyncio.sleep(0.1)
        print("Receiving SessionBegins ...")
        await _ws.recv()  # SessionBegins message
        print("Sending messages ...")
        
        async def send():
            while st.session_state['run']:
                try:
                    data = stream.read(FRAMES_PER_BUFFER)
                    data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"audio_data": data})
                    await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    if e.code == 4008:
                        break
                    else:
                        raise
                await asyncio.sleep(0.01)
        
        async def receive():
            while st.session_state['run']:
                try:
                    result_str = await _ws.recv()
                    result = json.loads(result_str)
                    if result["message_type"] == 'FinalTranscript':
                        st.markdown(result['text'])
                        st.session_state["audio_transcript"] = result["text"]
                        
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    if e.code == 4008:
                        break
                    else:
                        raise

        # Run send and receive concurrently
        await asyncio.gather(send(), receive())

# Streamlit App
def main():
    if 'run' not in st.session_state:
         st.session_state['run'] = False

    

    st.title("CRM Analysis for Pure Chem - Powered by Indra")

    # Input field for the question
    if 'audio_transcript' in st.session_state:
        question = st.text_input("Enter your question:", value=st.session_state.audio_transcript)
    else:
        question = st.text_input("Enter your question:")
    
    # Initialize session state
    if 'selected_key' not in st.session_state:
        st.session_state.selected_key = None
    if 'highlighted_button' not in st.session_state:
        st.session_state.highlighted_button = None

    st.write("Select an option to process the corresponding data:")

    # Create buttons in rows of 3
    num_buttons = len(button_names)
    num_columns = 3
    rows = (num_buttons + num_columns - 1) // num_columns

    for i in range(rows):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            idx = i * num_columns + j
            if idx < num_buttons:
                key = keys[idx]
                button_label = button_names[key]
                is_highlighted = (st.session_state.highlighted_button == key)

                # Create Streamlit button and handle click
                if cols[j].button(button_label, key=key, use_container_width=True):
                    st.session_state.selected_key = key
                    st.session_state.highlighted_button = key

    # Handle Reset button
    if st.button("Reset"):
        st.session_state.selected_key = None
        st.session_state.highlighted_button = None
        st.session_state.audio_transcript = ""
    
    # Record and transcribe audio
   # if st.button("Record Audio"):
       # audio_transcript = record_audio()
        #st.session_state.audio_transcript = audio_transcript

    # Start or stop real-time transcription
    if st.button('Start Listening'):
      st.session_state['run'] = True

    if st.button('Stop Listening'):
       st.session_state['run'] = False
      

    # Process question based on the selected button
    if st.session_state.selected_key:
        key = st.session_state.selected_key
        if question:
            result = process_question(question, key)
            st.write("### Answer:")
            st.write(result)
    if st.session_state['run']:
       asyncio.run(send_receive())         

if __name__ == "__main__":
    main()
