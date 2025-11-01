#!/usr/bin/env python3
"""
Simple Conversation class for interacting with Llama model via Ollama
"""

import ollama


class Conversation:
    """Manages conversation history with a Llama model"""

    def __init__(self, model='llama3.2', max_memory=-1):
        """
        Initialize a new conversation

        Args:
            model: The Ollama model to use (default: llama3.2)
            max_memory: Maximum number of messages to keep in history.
                       -1 means no limit. When exceeded, oldest messages are deleted.
        """
        self.model = model
        self.max_memory = max_memory
        self.messages = []

    def send(self, message, stream=False):
        """
        Send a message and get a response

        Args:
            message: The user message to send
            stream: If True, returns a generator that yields response chunks.
                   If False, returns the complete response as a string.

        Returns:
            str: The assistant's response (if stream=False)
            generator: A generator yielding response chunks (if stream=True)
        """
        # Add user message to history
        self.messages.append({
            'role': 'user',
            'content': message
        })

        if stream:
            return self._stream_response()
        else:
            return self._get_response()

    def _get_response(self):
        """Get complete response from model"""
        response = ollama.chat(
            model=self.model,
            messages=self.messages
        )

        assistant_message = response['message']['content']

        # Add assistant response to history
        self.messages.append({
            'role': 'assistant',
            'content': assistant_message
        })

        # Trim history if needed
        self._trim_history()

        return assistant_message

    def _stream_response(self):
        """Stream response from model (generator)"""
        stream = ollama.chat(
            model=self.model,
            messages=self.messages,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            yield content

        # Add complete response to history after streaming is done
        self.messages.append({
            'role': 'assistant',
            'content': full_response
        })

        # Trim history if needed
        self._trim_history()

    def _trim_history(self):
        """Trim message history to max_memory size if set"""
        if self.max_memory > 0 and len(self.messages) > self.max_memory:
            # Delete oldest messages to stay within limit
            excess = len(self.messages) - self.max_memory
            self.messages = self.messages[excess:]

    def clear(self):
        """Clear conversation history"""
        self.messages = []

    def get_history(self):
        """Get the conversation history"""
        return self.messages.copy()


# Example usage
if __name__ == "__main__":
    conv = Conversation()

    response = conv.send("What is 2+2?")
    print(f"Response: {response}\n")

    print(conv.send("What question did I just ask?"))
