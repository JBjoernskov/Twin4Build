from pathlib import Path
import re

def clean_rst_files(directory="source/auto"):
    """Clean up auto-generated RST files."""
    for file in Path(directory).glob("**/*.rst"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Get the module/package name from the first line
        first_line = content.split('\n')[0]
        name = first_line.split(' ')[0]
        
        # Remove " package" and " module" from titles while preserving the name
        if ' package' in first_line or ' module' in first_line:
            new_title = f"{name}\n{'=' * len(name)}\n"
            content = re.sub(r'^.*?\n=+\n', new_title, content, flags=re.MULTILINE)
        
        # Replace section headers
        content = content.replace("Subpackages\n-----------", "Package\n---------")
        content = content.replace("Submodules\n----------", "Module\n-------")
        
        # Add module title if it's missing (for submodules)
        if not re.search(r'^[^\n]+\n=+\n', content, re.MULTILINE):
            module_name = file.stem  # Get filename without extension
            if '.' in module_name:
                module_name = module_name.split('.')[-1]  # Get last part of module name
            content = f"{module_name}\n{'=' * len(module_name)}\n\n{content}"
        
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    clean_rst_files()