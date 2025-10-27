#Build with AI: AI-Powered Dashboards with Streamlit 

#Import packages
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
import altair as alt
import google.generativeai as genai

# Open file with Gemini API key
with open("../gemini_api_key.txt") as f:
    gemini_api_key = f.read().strip()

# Configure Gemini client
genai.configure(api_key=gemini_api_key)


#Configure page
st.set_page_config(page_title="Iris Dashboard", layout="wide")

#Write title
st.title("AI Powered Dashboard")

#Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

#Add sidebar filters
st.sidebar.header("Filter Options")
#Add species filter
species_options = st.sidebar.multiselect("Select species:", options=iris.target_names, default=list(iris.target_names))
#Allow users to change x-axis
x_axis = st.sidebar.selectbox("X-axis feature:", options=iris.feature_names, index=0)
#Allow users to change y-axis
y_axis = st.sidebar.selectbox("Y-axis feature:", options=iris.feature_names, index=1)

#Add chat widget in sidebar
st.subheader("Ask a question about Iris Dataset")
#Determine if chat history exists in the session state and initialize if it doesn't
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#Create text input field in sidebar to allow users to type in message
user_input = st.text_input("Type a message...", key="ui_input")
#Check if send button is clicked
if st.button("Send", key="ui_send"):
    if not user_input.strip():
        st.warning("Please enter a message before sending.")
    else:
        # Append user input to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Convert chat history to Gemini-compatible format
        gemini_history = []
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                gemini_history.append({"role": "user", "parts": [msg["content"]]})
            else:
                gemini_history.append({"role": "model", "parts": [msg["content"]]})

        # Add a system prompt at the start
        system_prompt = (
            "You are an expert on the Iris dataset and Python. "
            "If code is needed, reply only with a complete ```python``` block. "
            "The DataFrame is available as `df`. "
            "Columns are: 'sepal length (cm)', 'sepal width (cm)', "
            "'petal length (cm)', 'petal width (cm)', and 'species'. "
            "End code with an expression that evaluates to the result, no print or return statements."
        )

        msgs = [{"role": "user", "parts": [system_prompt]}] + gemini_history

        reply = None

        try:
            # Initialize Gemini model
            model = genai.GenerativeModel("models/gemini-2.5-flash")

            # Generate response (Gemini format)
            response = model.generate_content(msgs)

            # Extract text
            reply = response.text.strip() if response.text else ""
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        except Exception as e:
            error_msg = f"Bot: Error - {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            st.error(error_msg)

        # --- Process Gemini response ---
        if reply:
            if reply.strip().startswith("```python"):
                # Extract the code safely
                code = reply.strip().split("```python")[-1].split("```")[0]
                st.subheader("Generated Python Code")
                st.code(code, language="python")

                ns = {"pd": pd, "df": df, "iris": iris, "st": st, "alt": alt}

                try:
                    lines = [l for l in code.splitlines() if l.strip()]
                    if len(lines) == 1:
                        result = eval(lines[0], ns)
                    else:
                        *body, last = lines
                        exec("\n".join(body), ns)
                        result = eval(last, ns)

                    st.subheader("Execution Result")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error executing code: {e}")
            else:
                st.subheader("Answer")
                st.write(reply)

st.subheader("Chat Window")
#Loop through the chat history stored in session state and display each message
for message in st.session_state.chat_history:
    st.write(message)
    
#Filter DataFrame
filtered_df = df[df["species"].isin(species_options)]

#Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)

#Create scatter plot visualization
st.subheader("Scatter Plot")
scatter = (
    alt.Chart(filtered_df)
    .mark_circle(size=60)
    .encode(
        x=x_axis,
        y=y_axis,
        color="species",
        tooltip=iris.feature_names + ["species"]
    )
    .interactive()
)
st.altair_chart(scatter, use_container_width=True)

#Display summary statistics
st.subheader("Summary Statistics")
st.write(filtered_df.describe())

#Add dashboard footer
st.write("---")
st.write("Dashboard built with Streamlit and Altair")