import tkinter as tk
from tkinter import scrolledtext, messagebox
import requests
import json
import re
import time

class HealthAssistantChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("Health Assistant Chatbot")
        self.root.geometry("600x500")
        
        # Configure colors and fonts
        self.bg_color = "#f0f8ff"  # Light blue background
        self.user_color = "#e6f3ff"  # Light blue for user messages
        self.bot_color = "#ffffff"  # White for bot messages
        self.font = ("Arial", 12)
        self.last_api_call = 0  # For rate limiting
        
        self.setup_ui()
        self.setup_safety_checks()
        self.setup_local_knowledge()
        
    def setup_ui(self):
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=60, height=20,
            font=self.font, bg=self.bg_color
        )
        self.chat_display.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # User input area
        self.user_input = tk.Entry(
            self.root, width=50, font=self.font
        )
        self.user_input.pack(pady=5, padx=10, fill=tk.X)
        self.user_input.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = tk.Button(
            self.root, text="Send", command=self.send_message,
            bg="#1bdd4f", fg="white", font=self.font
        )
        self.send_button.pack(pady=5, padx=10, ipadx=10)
        
        # Add welcome message
        self.add_bot_message("Hello! I'm your health assistant. I can provide general health information. "
                            "Please remember I'm not a doctor. What would you like to know?")
    
    def setup_safety_checks(self):
        # Words/phrases that might indicate dangerous queries
        self.dangerous_keywords = [
            "take my life", "kill myself", "suicide", "self-harm", "self harm",
            "overdose", "poison", "dangerous dosage", "how to die", "end my life",
            "want to die", "don't want to live", "harm myself"
        ]
        
        # Medical conditions that require professional help
        self.serious_conditions = [
            "heart attack", "stroke", "seizure", "severe pain",
            "difficulty breathing", "unconscious", "heavy bleeding",
            "can't breathe", "chest pain", "fainted", "pass out", "paralysis",
            "severe burn", "choking", "broken bone"
        ]
        
        # Emergency response phrases
        self.emergency_responses = {
            "suicide": "I'm really sorry you're feeling this way. Please contact a mental health professional or call a suicide prevention hotline immediately. You're not alone.",
            "self-harm": "I'm concerned about your safety. Please reach out to someone you trust or contact a mental health professional right away.",
            "serious_condition": "This sounds serious. Please seek immediate medical attention or call emergency services."
        }
    
    def setup_local_knowledge(self):
        self.local_knowledge = {
            r"headache|head hurts": "Common headache causes include stress, dehydration, or eye strain. Try resting, drinking water, and taking a break from screens. If severe or persistent, consult a doctor.",
            r"fever|high temperature": "For adults, fever is generally 100.4°F (38°C) or higher. Rest and fluids are important. Contact a doctor if fever is very high or lasts more than 3 days.",
            r"cold|flu": "Typical symptoms include runny nose, cough, and fatigue. Get plenty of rest and fluids. See a doctor if symptoms worsen or persist.",
            r"back pain": "Mild back pain often improves with rest, gentle stretching, and over-the-counter pain relievers. Seek medical attention for severe or persistent pain.",
            r"stomach ache|stomach pain": "Common causes include indigestion, gas, or mild food intolerance. Try resting and drinking clear fluids. See a doctor if pain is severe or accompanied by vomiting.",
            r"cough": "A cough can be caused by colds, allergies, or irritation. Stay hydrated and consider honey for relief. Consult a doctor if it persists or worsens.",
            r"sore throat": "Often caused by viral infections. Gargle with warm salt water and stay hydrated. See a doctor if severe or lasting more than a week.",
            r"rash|skin irritation": "Keep the area clean and dry. Avoid scratching. See a doctor if the rash spreads, worsens, or is accompanied by fever.",
            r"allergy|allergic": "Avoid known allergens. Antihistamines may help with mild reactions. Seek emergency care for difficulty breathing or swelling.",
            r"sleep|insomnia": "Maintain a regular sleep schedule, limit screen time before bed, and create a restful environment. Consult a doctor if problems persist.",
             r"\b(thank you|thanks)\b": "You're welcome! Feel free to ask more health questions anytime.",
             r"\b(hi|hello|hey|greetings)\b": "Hello! I'm your health assistant. How can I help you today?",
             r"\b(how are you|what's up)\b": "I'm a virtual assistant here to provide general health information. What would you like to know?",
            
        }
    
    def add_user_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "You: " + message + "\n", "user")
        self.chat_display.tag_config("user", foreground="black", justify="right")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def add_bot_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "Assistant: " + message + "\n\n", "bot")
        self.chat_display.tag_config("bot", foreground="blue", justify="left")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self, event=None):
        user_query = self.user_input.get().strip()
        if not user_query:
            return
        
        self.add_user_message(user_query)
        self.user_input.delete(0, tk.END)
        
        # Check for dangerous queries first
        safety_response = self.check_safety(user_query)
        if safety_response:
            self.add_bot_message(safety_response)
            return
        
        # Check local knowledge base
        local_response = self.check_local_knowledge(user_query)
        if local_response:
            self.add_bot_message(local_response)
            return
        
        # Process the query with the LLM
        self.process_with_llm(user_query)
    
    def check_safety(self, query):
        query_lower = query.lower()
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in query_lower:
                return self.emergency_responses.get(keyword.split()[0], 
                    "Please seek immediate help from a professional.")
        
        # Check for serious medical conditions
        for condition in self.serious_conditions:
            if condition in query_lower:
                return self.emergency_responses.get("serious_condition")
        
        return None
    
    def check_local_knowledge(self, query):
        query_lower = query.lower()
        for pattern, response in self.local_knowledge.items():
            if re.search(pattern, query_lower):
                return response + "\n\nRemember: I'm not a doctor. For personal advice, please consult a healthcare professional."
        return None
    
    def process_with_llm(self, query):
        try:
            # Rate limiting - don't call API more than once every 2 seconds
            current_time = time.time()
            if current_time - self.last_api_call < 2:
                time.sleep(2 - (current_time - self.last_api_call))
            
            # Prepare the prompt with safety instructions
            prompt = f"""Act as a helpful medical assistant. Provide clear, friendly, and accurate information about general health topics. 
            Do not provide medical advice, diagnoses, or treatment recommendations. 
            Always remind users to consult with healthcare professionals for personal medical concerns.
            
            User question: {query}
            
            Assistant response:"""
            
            # Call Hugging Face Inference API
            api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
            headers = {
                "Authorization": "Bearer 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",  # Replace with your API key
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            # Show "typing" indicator
            self.root.after(500, lambda: self.chat_display.config(state=tk.NORMAL))
            self.root.after(500, lambda: self.chat_display.insert(tk.END, "Assistant is typing...\n", "typing"))
            self.root.after(500, lambda: self.chat_display.tag_config("typing", foreground="gray", justify="left"))
            self.root.after(500, lambda: self.chat_display.config(state=tk.DISABLED))
            self.root.after(500, lambda: self.chat_display.see(tk.END))
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
            # Remove typing indicator
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-2l", "end-1c")
            self.chat_display.config(state=tk.DISABLED)
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                # Extract only the assistant's response
                assistant_response = generated_text.split("Assistant response:")[-1].strip()
                # Clean up the response
                assistant_response = re.sub(r"<\/?s>", "", assistant_response)  # Remove any special tokens
                self.add_bot_message(assistant_response)
            else:
                self.add_bot_message("I couldn't generate a response for that. Could you try rephrasing your question?")
            
            self.last_api_call = time.time()
        
        except requests.exceptions.RequestException as e:
            # Remove typing indicator
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-2l", "end-1c")
            self.chat_display.config(state=tk.DISABLED)
            
            if "503" in str(e):
                self.add_bot_message("The model is currently loading. Please wait a moment and try again.")
            elif "timeout" in str(e).lower():
                self.add_bot_message("The request timed out. Please try again.")
            else:
                self.add_bot_message("I'm having trouble connecting to the knowledge base. Here's some general information that might help:\n\n" + 
                                    self.get_fallback_response(query))
        except Exception as e:
            # Remove typing indicator
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-2l", "end-1c")
            self.chat_display.config(state=tk.DISABLED)
            
            self.add_bot_message("An unexpected error occurred. Please try again later.")
    
    def get_fallback_response(self, query):
        """Provide a generic response when API fails"""
        health_topics = ["headache", "fever", "cold", "pain", "cough", "stomach", "rash"]
        query_lower = query.lower()
        
        for topic in health_topics:
            if topic in query_lower:
                return f"For {topic}-related concerns, it's often helpful to rest and stay hydrated. However, I recommend consulting a healthcare professional for proper advice."
        
        return "I couldn't retrieve specific information about your query. For health concerns, it's always best to consult with a healthcare professional."

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    chatbot = HealthAssistantChatbot(root)
    root.mainloop()