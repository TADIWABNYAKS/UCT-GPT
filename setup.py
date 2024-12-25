'''
Script to download gptNeo-3.7B and run a simple test prompt to confirm it has been setup 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_and_verify_model(model_name="EleutherAI/gpt-neo-2.7B", prompt="Are you working, bud?", max_length=50):
    try:
        print("Checking for model and tokenizer...")
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True)
        print("Model and tokenizer loaded successfully!")
        
        #Verify by generating text from model
        print("Generating test output...")
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        #Print generated text
        print("\nGenerated Text:")
        print(response)
        print("\nModel verification successful!")
        
    except Exception as e:
        print(f"Error during setup or verification: {e}")
        raise

if __name__ == "__main__":
    print("Starting setup and verification...")
    setup_and_verify_model()
    print("Setup and verification complete!")
