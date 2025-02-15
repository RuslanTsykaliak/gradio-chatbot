import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer using Hugging Face
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Explicitly load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Create the pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="pt")

def respond(
	message,
	history: list[tuple[str, str]],
	system_message,
	max_tokens,
	temperature,
	top_p,
):
	# Combine system message and conversation history
	prompt = system_message + "\n"
	prompt += f"User: {message}\n\nBot:"

	# Generate the response using the model
	response = chatbot(prompt, max_length=max_tokens, temperature=temperature, top_p=top_p)[0]['generated_text']
	return response

# Define the Gradio interface with additional inputs
demo = gr.ChatInterface(
	respond,
	additional_inputs=[
    	gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
    	gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
    	gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
    	gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
	],
)

if __name__ == "__main__":
	demo.launch()