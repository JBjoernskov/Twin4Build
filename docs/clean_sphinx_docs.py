# Standard library imports
import os
import re
from pathlib import Path


def clean_rst_files(directory="source/auto"):
    """Clean up auto-generated RST files."""
    for file in Path(directory).glob("**/*.rst"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        # Get the module/package name from the first line
        first_line = content.split("\n")[0]
        name = first_line.split(" ")[0]

        # Get just the last part of the module name for the title
        short_name = name.split(".")[-1]

        # Remove " package" and " module" from titles while preserving the short name
        if " package" in first_line or " module" in first_line:
            new_title = f"{short_name}\n{'=' * len(short_name)}\n"
            content = re.sub(r"^.*?\n=+\n", new_title, content, flags=re.MULTILINE)

        # Fix toctree indentation - only for actual toctree entries, not automodule directives
        # COMMENTED OUT: This was causing the indentation problem with discrete_statespace_system
        # lines = content.split('\n')
        # i = 0
        # while i < len(lines):
        #     line = lines[i]
        #     if line.strip().startswith('.. toctree::'):
        #         # Found a toctree directive, now look for its entries
        #         i += 1
        #         while i < len(lines):
        #             line = lines[i]
        #             stripped = line.strip()
        #
        #             # Skip empty lines and directive options (lines starting with :)
        #             if not stripped or stripped.startswith(':'):
        #                 i += 1
        #                 continue
        #
        #             # If we hit another directive, we're done with this toctree
        #             if stripped.startswith('.. '):
        #                 break
        #
        #             # If this line is not already indented and looks like a module reference
        #             # (contains dots but no spaces, and doesn't start with ..)
        #             if (not line.startswith('   ') and
        #                 '.' in stripped and
        #                 ' ' not in stripped and
        #                 not stripped.startswith('..')):
        #                 # This is a toctree entry that needs indentation
        #                 lines[i] = '   ' + line
        #
        #             i += 1
        #     else:
        #         i += 1
        # content = '\n'.join(lines)

        # Replace section headers
        content = content.replace("Subpackages\n-----------", "Package\n---------")
        content = content.replace("Submodules\n----------", "Modules\n-------")

        # Replace all long module names with just the last part
        # COMMENTED OUT: Complex regex patterns that were causing issues
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
        #
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
        # COMMENTED OUT: This was adding unnecessary titles
        # if not re.search(r'^[^\n]+\n=+\n', content, re.MULTILINE):
        #     module_name = file.stem  # Get filename without extension
        #     if '.' in module_name:
        #         module_name = module_name.split('.')[-1]  # Get last part of module name
        #     content = f"{module_name}\n{'=' * len(module_name)}\n\n{content}"

        # Clean single module sections
        # COMMENTED OUT: This was removing the "Modules" section header which breaks navigation
        # content = clean_single_module_sections(content)

        # Remove module contents section (this is redundant)
        content = remove_module_contents_section(content)

        # Write the modified content back to the file
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)


def clean_single_module_sections(content):
    # COMMENTED OUT: This function was removing the "Modules" section header
    # Remove "Modules" section header if followed by a single "xxx module" section
    # content = re.sub(
    #     r'(Modules\n[-=]+\n+)([^\n]+\n[-=]+\n+)',
    #     '',
    #     content,
    #     count=1
    # )
    return content


def remove_module_contents_section(content):
    # Remove "Module contents" section and its automodule directive
    # BUT preserve it for the main twin4build package to show the reader guide

    # First check if this is the main twin4build package file
    if (
        ".. automodule:: twin4build\n" in content
        and not ".. automodule:: twin4build." in content
    ):
        # This is the main twin4build package, change "Module contents" to "Note"
        content = content.replace("Module contents\n---------------", "Note\n----")
        return content

    # For all other modules, remove the Module contents section
    content = re.sub(
        r"Module contents\n[-=]+\n\n(\.\. automodule::[^\n]*\n(?:[ ]+:[^\n]*\n)*)",
        "",
        content,
        count=1,
    )
    return content


if __name__ == "__main__":
    # Standard library imports
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "source/auto"
    clean_rst_files(directory)
