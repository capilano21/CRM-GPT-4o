
---

# CRM Analysis using GPT 4o vision capabilities

## Overview

This Streamlit application integrates various functionalities to analyze CRM (Customer Relationship Management) data. It includes features for text-based queries, audio recording and transcription, and real-time transcription. The app utilizes OpenAI's language model for answering questions based on the provided data.

## Features

- **Text-Based Query Processing**: Enter questions to get answers based on pre-loaded data.
- **Audio Recording and Transcription**: Record audio and transcribe it to text.
- **Real-Time Transcription**: Stream audio for real-time transcription.

## Requirements

Ensure you have the following Python packages installed:
- `streamlit`
- `pickle`
- `langchain_openai`
- `langchain_core`
- `sounddevice`
- `scipy`
- `assemblyai`
- `websockets`
- `base64`
- `json`
- `pyaudio`

Install them using pip if you haven't already:

```bash
pip install streamlit pickle langchain_openai langchain_core sounddevice scipy assemblyai websockets pyaudio
```

## Setup

### API Keys

1. **OpenAI API Key**:Replace key in the script with your OpenAI API key.
2. **AssemblyAI API Key**: Replace key in the script with your AssemblyAI API key.
### Dictionary Files

The application requires two pickle files: `summary_dict.pkl` and `data_dict.pkl`. These files contain information needed for processing queries and answering questions.

#### Generating Dictionary Files

1. **Data Collection**: Collect and prepare your data. You should have data in a format where each key corresponds to a type of data (e.g., 'employee_records', 'customer_orders_pending').

2. **Creating Summary Dictionary**:
   - The `summary_dict` should be a dictionary where keys represent the data types and values provide a summary or description of what data is included in each type.

3. **Creating Data Dictionary**:
   - The `data_dict` should be a dictionary where keys correspond to the same data types as in `summary_dict`, and values contain the actual data.

4. **Pickle the Dictionaries**:
   Use the following Python script to pickle your dictionaries:

   ```python
   import pickle

   summary_dict = {
       'employee_records': 'Details about employee records.',
       'customer_orders_pending': 'Pending customer orders data.',
       'product_orders_pending': 'Pending product orders data.',
       # Add other entries as needed
   }

   data_dict = {
       'employee_records': 'Detailed employee records data...',
       'customer_orders_pending': 'Detailed pending customer orders data...',
       'product_orders_pending': 'Detailed pending product orders data...',
       # Add other entries as needed
   }

   with open('summary_dict.pkl', 'wb') as f:
       pickle.dump(summary_dict, f)

   with open('data_dict.pkl', 'wb') as f:
       pickle.dump(data_dict, f)
   ```

### Running the App

1. **Start the Application**:
   Navigate to the directory containing your script and run:

   ```bash
   streamlit run your_script_name.py
   ```

2. **Interact with the App**:
   - **Text-Based Query**: Enter a question and select the relevant data type to get an answer.
   - **Record Audio**: Use the recording feature to capture audio and transcribe it to text.
   - **Real-Time Transcription**: Start and stop real-time audio transcription as needed.

## Notes

- Ensure that you have set up the required API keys and dictionary files correctly.
- Adjust the `duration` in the `record_audio` function if you need longer or shorter recordings.
- The real-time transcription feature uses a websocket for AssemblyAI. Ensure your internet connection is stable for continuous streaming.

## Troubleshooting

- **API Key Errors**: Double-check your API keys and ensure they are correctly set in the script.
- **File Not Found**: Ensure that `summary_dict.pkl` and `data_dict.pkl` are in the same directory as the script.
- **Dependency Issues**: Ensure all required packages are installed and up-to-date.

---

Feel free to modify or extend this README as needed for your specific use case!
