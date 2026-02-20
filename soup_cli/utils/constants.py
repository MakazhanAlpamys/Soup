"""Global constants and defaults."""

APP_NAME = "soup"
CONFIG_FILE = "soup.yaml"
SOUP_DIR = ".soup"
EXPERIMENTS_DB = "experiments.db"

DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\\n' }}{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\\n' }}{% endif %}{% endfor %}"""
