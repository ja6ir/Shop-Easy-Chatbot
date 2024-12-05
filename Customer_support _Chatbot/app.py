import streamlit as st
import openai
import os


from lmstudio import LMStudioClient
from vector import search_vector_store, index, df, model

def clear_input():
    st.session_state["input"] = ""

def main():
    st.title("ShopEasy Chatbot")
    st.write("This chatbot interface interacts with a model running on LM Studio.")

    # Initialize conversation context in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a customer support chatbot of ShopEasy for resolving customer queries ."}
        ]

    # Sidebar options for parameters
    st.sidebar.title("Configuration")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=-1, value=-1)

    user_input = st.text_input("You:", key="input")
    
    chat_history_container = st.empty()
    conversation_container = st.empty()

    retrieved_text = search_vector_store(user_input,index,model,df, top_k=5)

    if st.button("Send") and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Create LMStudio client and send request
        client = LMStudioClient()
        stream = client.send_request(
            st.session_state.messages,
            retrieved_text= retrieved_text,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        if "chat_response" in st.session_state:
            st.session_state.chat_response = f"\n\n**User:** {user_input}\n\n" + st.session_state.chat_response
        else:
            st.session_state.chat_response=f"\n\n**User:** {user_input}\n\n" 

        conversation_container.markdown(st.session_state.chat_response)
        
        response = "**Assistant:** "

        # Iterate through the streaming response and update the frontend
        for snippet in stream:
            if snippet.strip():  # Ensure the snippet is not just whitespace
                # Add the snippet to the chat history
                
                response += snippet
               
                
                # Clear the chat history container
                chat_history_container.empty()

                # Update UI elements with the new content using Markdown
                chat_history_container.markdown(response)
        st.session_state.chat_response = f"{response}  "+ st.session_state.chat_response

if __name__ == "__main__":
    main()