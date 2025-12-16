"""
Gmail LangChain Integration Package

This package provides Gmail functionality using the LangChain Community toolkit
for the cookie delivery agent system.
"""

from .gmail_manager import LangChainGmailManager, gmail_manager, LANGCHAIN_GMAIL_AVAILABLE
from .email_utils import send_confirmation_email_langchain, search_gmail_messages, get_gmail_message_details

__all__ = [
    'LangChainGmailManager',
    'gmail_manager', 
    'LANGCHAIN_GMAIL_AVAILABLE',
    'send_confirmation_email_langchain',
    'search_gmail_messages',
    'get_gmail_message_details'
]
