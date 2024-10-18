import os
import faiss
import pandas as pd
import numpy as np
import streamlit as st
from langchain.schema.runnable import Runnable
import google.generativeai as genai
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import faiss
from langchain.prompts import load_prompt
from streamlit import session_state as ss
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import uuid
import json
import time
import datetime

from helper_functions.faiss_retriever import faiss_retriever

# def faiss_retriever(index, query_vector, df, k=5):
#     query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
    
#     # Perform search
#     distances, indices = index.search(query_vector, k)
    
#     # Filter by score_threshold
#     results = []
#     for dist, idx in zip(distances[0], indices[0]):   
#         result = {
#             "sentence_chunk": df.iloc[idx]["sentence_chunk"],
#             "page_number": df.iloc[idx]["page_number"],
#             "distance": dist
#         }
#         results.append(result)

#     return results

# Was trying to integrate the Gemini API with the LangChain framework but its not required since I have implemented the RetrievalChain from scratch
class GenerativeModelWrapper(Runnable):
    def __init__(self, model: genai.GenerativeModel):
        self.model = model

    def _run(self, inputs):
        # Assuming inputs is a dictionary with a 'prompt' key
        response = self.model.generate_content(inputs['prompt']) 
        return response

    async def _arun(self, inputs):
        # If you want to make it asynchronous
        pass

    def invoke(self, inputs):
        return self._run(inputs)
# Example of how to use the GenerativeModelWrapper with the Gemini model
# response = wrapped_model.invoke({'prompt': prompt})

# Same as before, but with the addition of the vectorstore argument (not required for the current implementation)
class CustomFaissRetriever(BaseModel):
    index: any = Field(...)
    df: any = Field(...)
    #vectorstore: any = Field(...)  # Add this line

    class Config:
        arbitrary_types_allowed = True  # Allow FAISS index and DataFrame types

    def _get_relevant_documents(self, query: str):
        # Get query embedding using Gemini API
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_document",
            title="profile"
        )["embedding"]

        # Retrieve relevant documents using FAISS retriever
        results = faiss_retriever(self.index, query_embedding, self.df)

        # Convert FAISS results to LangChain Document objects
        docs = [
            Document(
                page_content=res["sentence_chunk"],
                metadata={"page_number": res["page_number"], "distance": res["distance"]}
            )
            for res in results
        ]
        return docs


def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

# if "mongodB_pass" in os.environ:
#     mongodB_pass = os.getenv("mongodB_pass")
# else: mongodB_pass = st.secrets["mongodB_pass"]
mongodB_pass = os.getenv("mongodB_pass") or st.secrets["mongodb"]["mongodB_pass"]
gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets["mongodb"]["gemini_api_key"]
icon_path = "assets/illustrator.png"
# Setting up a mongo_db connection to stosre conversations for deeper analysis
uri = f"mongodb+srv://sauravm:{mongodB_pass}@geminidb.ionht.mongodb.net/?retryWrites=true&w=majority&appName=geminiDB"

# Ensure MongoDB connection is initialized only once without caching issues
@st.cache_resource(show_spinner=False)
def init_connection():
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

client = init_connection()
if client is None:
    st.stop()  # Safely stop if the connection fails


# Set the API key for the Gemini model
genai.configure(api_key=gemini_api_key)
# Initialize the Gemini model
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                     generation_config=genai.GenerationConfig(temperature=0.6))

# Wrap the Gemini model in the Runnable
wrapped_model = GenerativeModelWrapper(gemini_model)

# # Caching the connection to the database
# @st.cache_resource
# def init_connection():
#     return MongoClient(uri, server_api=ServerApi('1'))
# client = init_connection()

# Setting up the database and collection
db = client['conversations_db']
conversations_collection = db['conversations']

# if "GEMINI_API_KEY" in os.environ:
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
# else: gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Change the title of the Streamlit app
st.set_page_config(page_title="Saurav AI", page_icon= icon_path, layout="centered", initial_sidebar_state="auto")

# Creating the sidebar
with st.sidebar:
    # Profile picture
    st.markdown(
        """
        <style>
            .profile-container {
                display: flex; /* Use flexbox for centering */
                justify-content: center; /* Center horizontally */
                align-items: center; /* Center vertically */
                height: 100%; /* Full height to center vertically */
                margin: 20px;
            }

            .profile-picture {
                border-radius: 50%;
                width: 170px;
                height: 170px;
                transition: transform 0.3s, box-shadow 0.3s;
                border: 5px solid #ffffff; /* White border for visibility */
            }

            .profile-container:hover .profile-picture {
                transform: scale(1.1); /* Slightly enlarge on hover */
                box-shadow: 0 0 20px rgba(255, 255, 255, 0.7); /* Glow effect on hover */
            }
        </style>

        <div class="profile-container">
            <img class="profile-picture" src="https://lh3.googleusercontent.com/a/ACg8ocIm5OvVulFt5K4RL7mQ22E-KToGqr1pJrGEAivi3dNn0ZLedm4h=s432-c-no" alt="Profile Picture" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # st.markdown(
    #     """
    #     <div style="text-align: center;">
    #         <img src="https://lh3.googleusercontent.com/a/ACg8ocIm5OvVulFt5K4RL7mQ22E-KToGqr1pJrGEAivi3dNn0ZLedm4h=s432-c-no" alt="Profile Picture" style="border-radius:50%; width:170px; height:170px;" />
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )
    #"https://sauravmestry.netlify.app/_next/image?url=%2Fprofile.jpg&w=828&q=75"

    
    st.markdown(
        """
        <div style="text-align: center; padding-top: 10px;">
            <p style="margin-bottom: 0px; font-size: 25px; font-style: bold;">Saurav Sharad Mestry</p>
            <p style="margin-bottom: 5px;  font-style: italic; font-size: 16px; color: #cccccc;">
                Software Developer | Data Engineer
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Small Paragraph about yourself
    st.markdown(
        """
        <style>
            .cool-text {
                margin-bottom: 0;
                font-style: italic;
                color: #cccccc;
                position: relative;
                display: inline-block;
                overflow: hidden;
                font-size: 16px;
                animation: pulse 1.5s infinite;
                animation-timing-function: ease-in-out;
            }

            .cool-text::before {
                content: attr(data-text);
                position: absolute;
                left: 0;
                top: 0;
                color: #ffffff;
                overflow: hidden;
                white-space: nowrap;
                width: 0;
                animation: reveal 5s forwards;
                transition: all 0.3s ease;
                text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
            }

            @keyframes reveal {
                to {
                    width: 100%;
                }
            }
        </style>
        <div style="text-align: justify; padding: 10px;">
            <p style="margin-bottom: 10px; font-size: 16px; color: #ffffff;">
                I am a Software Developer and Data Engineer with 2.7 years of experience in developing microservices, big data pipelines, and machine learning. I am proficient in Python, Java, and their respective frameworks.
            </p>
            <p class="cool-text" data-text="Connect with me on the below links.">Connect with me on the below links</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Create a 2x2 table layout using HTML with inline CSS for consistent size
    st.markdown(
        """
        <style>
    .social-table {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
    }

    .social-table td {
        text-align: center;
        padding: 15px;
    }

    .social-button {
        display: inline-block;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .social-button:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
</style>

<table class="social-table">
    <tr>
        <td>
            <a class="social-button" href="https://www.linkedin.com/in/saurav-mestry/" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
            </a>
        </td>
        <td>
            <a class="social-button" href="https://www.github.com/CoolboySaurav" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
            </a>
        </td>
    </tr>
    <tr>
        <td>
            <a class="social-button" href="https://sauravmestry.netlify.app/" target="_blank">
                <img src="https://img.shields.io/badge/Website-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Website" />
            </a>
        </td>
        <td>
            <a class="social-button" href="mailto:sauravm@arizona.edu" target="_blank">
                <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
            </a>
        </td>
    </tr>
</table>


        """,
        unsafe_allow_html=True,
    )
    # st.markdown(
    #     """
    #     <table style="width:100%; table-layout:fixed;">
    #         <tr>
    #             <td style="text-align:center; padding:10px;">
    #                 <a href="https://www.linkedin.com/in/saurav-mestry/" target="_blank">
    #                     <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
    #                 </a>
    #             </td>
    #             <td style="text-align:center; padding:10px;">
    #                 <a href="https://www.github.com/CoolboySaurav" target="_blank">
    #                     <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
    #                 </a>
    #             </td>
    #         </tr>
    #         <tr>
    #             <td style="text-align:center; padding:10px;">
    #                 <a href="https://sauravmestry.netlify.app/" target="_blank">
    #                     <img src="https://img.shields.io/badge/Website-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white" />
    #                 </a>
    #             </td>
    #             <td style="text-align:center; padding:10px;">
    #                 <a href="mailto:sauravm@arizona.edu" target="_blank">
    #                     <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
    #                 </a>
    #             </td>
    #         </tr>
    #     </table>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saurav-mestry/)")
    # st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://www.github.com/CoolboySaurav)")
    # st.markdown("[![Website](https://img.shields.io/badge/Website-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://sauravmestry.netlify.app/)")
    # st.markdown("[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sauravm@arizona.edu)")
    # st.markdown("[![Resume](https://img.shields.io/badge/Resume-4285F4?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1qVYQh3Z7L0J2Ym7R4j0bH3sJXH6J6H9s/view?usp=sharing)")

    # Display current time
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 10px; color: gray; font-style: italic;">
            <p>{time.strftime("%A, %d %B %Y")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    #st.markdown(time.strftime("%A, %d %B %Y"))


    #st.link_button("Fork me on GitHub", "https://img.shields.io/badge/Fork_on_GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CoolboySaurav/streamlit-app")  # Update with your GitHub repository link



#Creating Streamlit title and adding additional information about the bot
#st.title("Saurav AI 1.0")

# Updated HTML/CSS for an Animated Title with Improved Design
html_code = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@500&family=Orbitron:wght@700&display=swap');

    /* Container for Title */
    .custom-title-container {
        text-align: center;
        margin-top: -20px;
        margin-bottom: 20px;
    }

    /* Main Title with Neon Glow */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5em;
        font-weight: 700;
        color: #00FFFF;
        text-shadow: 0 0 15px #00FFFF, 0 0 30px #00FFFF, 0 0 45px #00E6E6, 0 0 60px #00B2B2;
        animation: neon-flicker 2s infinite alternate;
        margin: 0;
    }

    /* Subtitle with Smooth Typing Animation */
    .subtitle {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.2em;
        color: #b3b3b3;
        overflow: hidden;
        white-space: nowrap;
        border-right: 2px solid #b3b3b3;
        animation: typing 4s steps(30, end), blink-cursor 0.75s step-end infinite;
        margin-top: 10px;
    }

    /* Neon Flicker Effect */
    @keyframes neon-flicker {
        0% { text-shadow: 0 0 20px #00FFFF, 0 0 40px #00FFFF; }
        50% { text-shadow: 0 0 15px #00FFFF, 0 0 30px #00B2B2; }
        100% { text-shadow: 0 0 25px #00E6E6, 0 0 50px #00FFFF; }
    }

    /* Typing Animation for Subtitle */
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }

    @keyframes blink-cursor {
        50% { border-color: transparent; }
    }

    /* Link Styles to Prevent Underline */
    a {
        text-decoration: none;
    }
</style>

<!-- HTML Block -->
<div class="custom-title-container">
    <h1>
        <h1 class="main-title">Ask Saurav</h1>
    </h1>
    <div class="subtitle">Everything About Me, Simplified</div>
</div>
"""

# Render the title with CSS animations in Streamlit
st.markdown(html_code, unsafe_allow_html=True)

with st.expander("⚠️Disclaimer"):
    st.write("""**Hello!** I am Saurav AI 1.0, a chatbot created by **Saurav Mestry**. My primary function is to provide information about Saurav Mestry, his background, skills, and projects. For business inquiries or collaboration opportunities, please reach out to Saurav on [LinkedIn](https://www.linkedin.com/in/saurav-mestry/). 

    Please note the following:
    - The responses provided by me are based on pre-fed data and publicly available information.
    - I may not always have the latest updates. For time-sensitive or specific inquiries, please contact Saurav directly.
    - Any opinions or advice given are not legally binding or official statements from Saurav Mestry.

    Privacy: I do not store or process personal data, and all conversations are session-based. For detailed inquiries, it is recommended to contact Saurav through official channels.

    Thank you for interacting with Saurav AI!""")

path = os.getcwd()

# Loading prompt to query Gemini-1.5-flash model
prompt_template = path+"/templates/template.json"
prompt = load_prompt(prompt_template)

# Loading embedings
faiss_index = path+"/faiss_index.bin"

# Loading all the data files 
data_source = path + "/data/embeddings.csv"
pdf_source = path + "data/resume.pdf"

# Function to store conversation
def store_conversation(conversation_id, user_message, bot_message, answered):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "user_message": user_message,
        "bot_message": bot_message,
        "answered": answered
    }
    conversations_collection.insert_one(data)

# Load the FAISS index
faiss_index = faiss.read_index(faiss_index)
embeddings_dataframe = pd.read_csv(data_source)

# Step 4: Create a Prompt Template
prompt = """"System: You are Saurav Sharad Mestry and pretend as Saurav Mestry is talking when you ask anything, a comprehensive, interactive resource for exploring Saurav's background, skills, and expertise. Be polite and provide answers based on the provided context only as I. Do not start with "Hi! I'm Saurav" for every query unless asked to. Use only the provided data and not prior knowledge. \n Human: Take a deep breath and do the following step by step these 4 steps: \n 1. Read the context below \n 2. Answer the question using only the provided Help Centre information \n 3. Make sure to nicely format the output so it is easy to read on a small screen. \n4. Provide 3 examples of questions user can ask about me (Saurav Mestry) based on the questions from context and chat_history. Chat_History : \n ---{chat_history}--- \n Context : \n ~~~ {context} ~~~ \n User Question: --- {question} --- \n \n If a question is directed at you, clarify that you are Saurav and proceed to answer as if the question were addressed to Saurav Mestry and answer as I. If you lack the necessary information to respond, simply state that you don't know; do not fabricate an answer. If a query isn't related to Saurav Mestry's background, politely indicate that you're programmed to answer questions solely about his experience, education, training, and aspirations. Offer three sample questions users could ask about Saurav Mestry for further clarity. When responding, aim for detail but limit your answer to a maximum of 150 words. Ensure your response is formatted for easy reading. Your output should be in a json format with 3 keys: answered - type boolean, response - markdown of your answer, questions - list of 3 suggested questions. Ensure your response is formatted for easy reading and please use only context to answer the question - my job depends on it. \n\n ```json"""

prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"], 
    template=prompt
)

# Initialize the custom FAISS retriever
retriever = CustomFaissRetriever(index=faiss_index, df=embeddings_dataframe)  # Add vectorstore argument

def conversational_chat(query: str):
   with st.spinner("Thinking..."):
      
        # Retrieve relevant documents using the custom FAISS retriever
        docs = retriever._get_relevant_documents(query)

        # Extract the chat history and context
        chat_history = "\n".join(
            [f"user_message: {q}\n bot_message: {a}" for q, a in st.session_state['history']]
        )

        if not docs:
            context = "I don't have an answer to that question. Can you ask me something else?"
        else:
            context = "\n".join([doc.page_content for doc in docs])
        
        # Generate the final prompt using the retrieved context
        final_prompt = prompt_template.format(context=context, question=query, chat_history=chat_history)

        # Use the wrapped model to generate a response
        result = wrapped_model.invoke({'prompt': final_prompt})
        # Extract the text content from the response
        result_text = result.candidates[0].content.parts[0].text

        # Remove the JSON code block markers
        json_string = result_text.replace("```json", "").replace("```", "").strip()

        # Parse the JSON string
        result = json.loads(json_string)

        # Scrap code for handling the case where the response is not satisfactory
        # Ensure valid JSON response handling
        # if result["answered"]:
        #     data = result
        # else:
        #     data = json.loads('''{
        #         "answered": "false",
        #         "response": "Hmm... Something is not right. I'm experiencing technical difficulties. Try asking your question again.",
        #         "questions": [
        #             "What is Saurav's professional experience?",
        #             "What projects have Saurav has worked on?",
        #             "What are Saurav's career goals?"
        #         ]
        #     }''')

        data = result
        # Extract the response details
        answered = data.get("answered")
        response = data.get("response")
        questions = data.get("questions")
        
        print(answered)
        print(response)
        print(context)
        
        full_response = ""
        # Store the chat history
        st.session_state['history'].append((query, response))

        # Handle cases where response is empty or not satisfactory

        markdown_list = "\n".join([f"- {item}" for item in questions])
        
        # Removing self-introduction from the response if it is not the first response
        if len(st.session_state['history']) > 1:
            index = -1
            index = response.find("Saurav")
            if index != -1:
                index2 = response.find("Mestry")
                index = max(index, index2)

            # Removing everything before the first occurrence of "Mestry" or "Saurav" if none of them are present then dont do anything
            if index != -1:
                # Find the next "." and move the index to the next character
                index = response.find(".", index)
                response = response[index+1:]


        if answered:
            full_response = f"{response}\n\n What else would you like to know? You can ask:\n{markdown_list}"
            store_conversation(st.session_state["uuid"], query, full_response, answered)
        else:
            full_response = f"{response}\n\n I can only provide you information about my professional background and qualifications. If you have other inquiries, I recommend reaching out to me on [LinkedIn](https://www.linkedin.com/in/saurav-mestry/). I can answer questions like:\n{markdown_list}"
            store_conversation(st.session_state["uuid"], query, full_response, answered)
        
        
        # Scrap code for handling the case where the response is not satisfactory
        # if 'I am tuned to only answer questions' in response or response.strip() == "" or response == "Hmm... Something is not right. I'm experiencing technical difficulties. Try asking your question again.":
        #     full_response = """ Unfortunately, I can't answer this question. I can only provide you information about my professional background and qualifications. If you have other inquiries, I recommend reaching out to me on [LinkedIn](https://www.linkedin.com/in/saurav-mestry/). I can answer questions like: \n - What is your educational background?\n - Can you list your professional experience?\n - What skills do you possess?\n"""
        #     store_conversation(st.session_state["uuid"], query, full_response, answered)
        # else:
        #     # Create a list of follow-up questions
        #     markdown_list = "\n".join([f"- {item}" for item in questions])
        #     full_response = (
        #         f"{response}\n\nWhat else would you like to know? You can ask:\n{markdown_list}"
        #     )
        #     store_conversation(st.session_state["uuid"], query, full_response, answered)

        return full_response


# Ensure that each session has a unique identifier
if "uuid" not in st.session_state:
    st.session_state["uuid"] = str(uuid.uuid4())

# Setting up the history of the chat
if "history" not in st.session_state:
    st.session_state["history"] = []  # Store chat history


if "message" not in st.session_state:
    st.session_state["message"] = []

    with st.chat_message("assistant"):
        
        message_placeholder = st.empty()

        welcome_message = "Hello! Namaste! Hola! I’m Saurav AI, an intelligent chatbot developed by Saurav Mestry. I'm here to provide insights and answer any questions you have about him. Feel free to ask me anything!"
        message_placeholder.markdown(welcome_message)

for message in st.session_state.message:
    with st.chat_message(message["role"]):

        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about Saurav Mestry"):
    st.session_state.message.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        user_input = prompt
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        full_response = conversational_chat(user_input)
        message_placeholder.markdown(full_response)
    
    st.session_state.message.append({"role": "assistant", "content": full_response})