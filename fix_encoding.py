"""
Quick script to remove emojis from print statements
"""
import re
from pathlib import Path

files_to_fix = [
    "src/config.py",
    "src/vector_store.py",
    "src/rag_chain.py",
    "src/document_loader.py"
]

# Common emojis to remove
emojis = ['📂', '⚠️', '❌', '✅', '🔢', '📊', '💾', '🔍', '📄', '✂️', '📋', '🧪', '🤖', '💬', '📚']

for file_path in files_to_fix:
    path = Path(file_path)
    if not path.exists():
        print(f"Skipping {file_path} - not found")
        continue

    print(f"Fixing {file_path}...")

    # Read file
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove emojis
    for emoji in emojis:
        content = content.replace(emoji + ' ', '')
        content = content.replace(emoji, '')

    # Write back
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ Fixed {file_path}")

print("\nAll files fixed!")
