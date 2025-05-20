from pathlib import Path
import re
import os


def clean_rst_files(directory="source/auto"):
    """Clean up auto-generated RST files."""
    for file in Path(directory).glob("**/*.rst"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Get the module/package name from the first line
        first_line = content.split('\n')[0]
        name = first_line.split(' ')[0]
        
        # Get just the last part of the module name for the title
        short_name = name.split('.')[-1]
        
        # Remove " package" and " module" from titles while preserving the short name
        if ' package' in first_line or ' module' in first_line:
            new_title = f"{short_name}\n{'=' * len(short_name)}\n"
            content = re.sub(r'^.*?\n=+\n', new_title, content, flags=re.MULTILINE)
        
        # Fix toctree indentation
        content = re.sub(
            r'(\.\. toctree::.*?\n)([^\s])',
            r'\1   \2',
            content,
            flags=re.DOTALL
        )
        
        # Replace section headers
        content = content.replace("Subpackages\n-----------", "Package\n---------")
        content = content.replace("Submodules\n----------", "Modules\n-------")
        
        # Replace all long module names with just the last part
        # patterns = [
        #     # For module sections with underlines
        #     r'([\w\.]+\.)([^\s]+)( module\n-+\n)',
        #     # For automodule directives
        #     r'(\.\. automodule:: [\w\.]+\.)([^\s]+)',
        #     # For toctree entries
        #     r'   [\w\.]+\.([^\s]+)',
        #     # For module headers
        #     r'[\w\.]+\.([^\s]+)( module)',
        #     # For any remaining long module names
        #     r'[\w\.]+\.([^\s]+)(?= (?:module|package))'
        # ]
        
        # for pattern in patterns:
        #     if 'toctree' in pattern:
        #         content = re.sub(pattern, r'   \1', content)
        #     else:
        #         content = re.sub(
        #             pattern,
        #             lambda m: f"{m.group(1)}{m.group(2) if len(m.groups()) > 1 else ''}" if len(m.groups()) > 1 else m.group(1),
        #             content
        #         )
        
        # Add module title if it's missing (for submodules)
        if not re.search(r'^[^\n]+\n=+\n', content, re.MULTILINE):
            module_name = file.stem  # Get filename without extension
            if '.' in module_name:
                module_name = module_name.split('.')[-1]  # Get last part of module name
            content = f"{module_name}\n{'=' * len(module_name)}\n\n{content}"
        
        # Clean single module sections
        content = clean_single_module_sections(content)
        
        # Remove module contents section
        content = remove_module_contents_section(content)
        
        # Write the modified content back to the file
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)

def clean_single_module_sections(content):
    # Remove "Modules" section header if followed by a single "xxx module" section
    content = re.sub(
        r'(Modules\n[-=]+\n+)([^\n]+\n[-=]+\n+)',
        '',
        content,
        count=1
    )
    return content

def remove_module_contents_section(content):
    # Remove "Module contents" section and its automodule directive
    content = re.sub(
        r'Module contents\n[-=]+\n\n(\.\. automodule::[^\n]*\n(?:[ ]+:[^\n]*\n)*)',
        '',
        content,
        count=1
    )
    return content

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "source/auto"
    clean_rst_files(directory)