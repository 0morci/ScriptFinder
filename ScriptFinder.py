import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import unicodedata

# Load the Sentence Transformer model
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_model()

def get_script_url_from_chatgpt(movie_title):
    prompt = f"Given the movie title '{movie_title}', provide the exact URL for its script on https://imsdb.com/. Only return the URL, nothing else."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides URLs for movie scripts on imsdb.com."},
            {"role": "user", "content": prompt}
        ]
    )
    
    content = response.choices[0].message.content.strip()
    
    # Use regex to extract URL if it's embedded in text
    url_match = re.search(r'https?://(?:www\.)?imsdb\.com/scripts/[A-Za-z0-9-]+\.html', content)
    if url_match:
        return url_match.group(0)
    elif content.startswith("https://imsdb.com/"):
        return content
    else:
        return None

def get_script_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            pre_tag = soup.find('pre')
            if pre_tag:
                return pre_tag.get_text()
    except requests.RequestException:
        pass
    return None

def search_script(question, script_content):
    # Split the script into chunks
    chunks = script_content.split('\n\n')
    
    # Encode the question and chunks
    question_embedding = sentence_model.encode([question])[0]
    chunk_embeddings = sentence_model.encode(chunks)
    
    # Calculate cosine similarity
    similarities = np.dot(chunk_embeddings, question_embedding) / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding))
    
    # Get the top 20 most relevant chunks
    top_indices = np.argsort(similarities)[-20:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    
    return ' '.join(relevant_chunks)

def truncate_text(text, max_tokens=3000):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

def answer_question(question, context, movie_title):
    truncated_context = truncate_text(context)
    
    # First, check for important characters
    character_check_prompt = f"In the movie '{movie_title}', is there a character that fits this description: {question}? If yes, provide brief factual information about the character but don't include the movie title in your response or Yes/No."
    
    character_check_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a movie expert. Provide factual information about characters if they exist in the movie. Make the response concise and direct."},
            {"role": "user", "content": character_check_prompt}
        ],
        max_tokens=100
    )
    
    character_info = character_check_response.choices[0].message.content.strip()
    
    if "there is no character" not in character_info.lower():
        return f"{character_info}"
    
    # If no character is found, proceed with the original context-based answer
    prompt = f"Based on the following context from the movie '{movie_title}', answer the question.\n\nContext: {truncated_context}\n\nQuestion: {question}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about movie scripts. Make the response concise and direct."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    answer = response.choices[0].message.content.strip()
    
    # Fact-checking step
    fact_check_prompt = f"Fact check the following answer about the movie '{movie_title}':\n\nQuestion: {question}\nAnswer: {answer}\n\nIs this answer correct and complete? If not, provide the correct and complete answer."
    
    fact_check_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a movie fact checker. Verify the given answer and correct or complete it if necessary. Make the response concise and direct."},
            {"role": "user", "content": fact_check_prompt}
        ],
        max_tokens=150
    )
    
    fact_check_result = fact_check_response.choices[0].message.content.strip()
    
    return f"Based on the movie '{movie_title}', {fact_check_result}"

def clean_text(text):
    # Remove control characters and non-printable characters
    cleaned = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' and ch.isprintable())
    # Replace any remaining problematic characters with spaces
    cleaned = ''.join(ch if ord(ch) < 128 else ' ' for ch in cleaned)
    return cleaned

def create_pdf(text, title):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    def add_page():
        c.setFont("Helvetica", 12)
        return c.beginText(50, height - 100)
    
    text_object = add_page()
    text_object.setFont("Courier", 10)
    
    # Replace tabs with spaces and handle end of line characters
    cleaned_text = text.replace('\t', '    ').replace('\n', '\n ')
    lines = cleaned_text.split('\n')
    for line in lines:
        try:
            text_object.textLine(line.rstrip())
        except UnicodeEncodeError:
            cleaned_line = ''.join(char for char in line if ord(char) < 128)
            text_object.textLine(cleaned_line.rstrip())
        
        if text_object.getY() < 50:  # Check if we're near the bottom of the page
            c.drawText(text_object)
            c.showPage()
            text_object = add_page()
            text_object.setFont("Courier", 10)  # Set font for new page
    
    c.drawText(text_object)
    c.save()
    
    buffer.seek(0)
    return buffer

def movie_script_agent():
    st.title("ScriptFinder 🍿")
    st.write("Enter a movie title to search for its script on IMSDb.")
# Set up OpenAI client

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'script_content' not in st.session_state:
        st.session_state.script_content = None
    if 'current_movie' not in st.session_state:
        st.session_state.current_movie = None

    # Function to check if the provided API key is valid
    def is_api_key_valid(api_key):
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        # Making a simple API call to the OpenAI endpoint
        try:
            response = requests.get("https://api.openai.com/v1/models", headers=headers)
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error occurred while validating API key: {e}")
            return False


    # Input field for user to enter their OpenAI API key
    if "user_api_key" not in st.session_state:
        st.session_state.user_api_key = None

    user_api_key = st.text_input("Enter your OpenAI API key:", type="password")

    


    # Validate the API key upon entry
    if user_api_key:
        if is_api_key_valid(user_api_key):
            st.session_state.user_api_key = user_api_key
            st.success("API key is valid!")
            global client
            client = OpenAI(api_key=user_api_key)
        else:
            st.error("Invalid API key. Please check and try again.")

    # Ensure user has entered their API key
    if not st.session_state.user_api_key:
        st.warning("Please enter your API key to continue.")
        st.stop()
    movie_title = st.text_input("Enter a movie title:")

    if st.button("Search"):
        if movie_title:
            script_url = get_script_url_from_chatgpt(movie_title)
            
            if script_url:
                st.success(f"Script URL found: {script_url}")
                script_content = get_script_content(script_url)
                if script_content:
                    st.session_state.script_content = script_content
                    st.session_state.current_movie = movie_title
                else:
                    st.error("Sorry, couldn't retrieve the script content.")
            else:
                st.error(f"Sorry, couldn't find a script for '{movie_title}' on IMSDb.")

    if st.session_state.script_content:
        st.write("Here's a preview of the script (first 500 characters):")
        st.text(st.session_state.script_content[:500] + "...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pdf_buffer = create_pdf(st.session_state.script_content, st.session_state.current_movie)
            st.download_button(
                label="Download Script (PDF)",
                data=pdf_buffer,
                file_name=f"{st.session_state.current_movie.replace(' ', '_')}_script.pdf",
                mime="application/pdf"
            )
        
        with col3:
            if st.button("New Search"):
                st.session_state.script_content = None
                st.session_state.current_movie = None
                st.session_state.conversation = []
                st.rerun()

        if st.session_state.conversation or st.session_state.script_content:
            for message in st.session_state.conversation:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            question = st.chat_input("Ask anything about the movie!")
            if question:
                st.session_state.conversation.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    relevant_context = search_script(question, st.session_state.script_content)
                    answer = answer_question(question, relevant_context, st.session_state.current_movie)
                    message_placeholder.markdown(answer)
                st.session_state.conversation.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    movie_script_agent()
