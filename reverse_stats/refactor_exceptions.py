import os
import glob
import re

files = glob.glob("*.py")
for f_path in files:
    if f_path in ["exceptions.py", "pipeline_types.py", "refactor_exceptions.py"]:
        continue
    
    with open(f_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if file has an Exception based error
    if re.search(r'class (\w+Error)\(Exception\):', content):
        content = re.sub(r'class (\w+Error)\(Exception\):', r'class \1(ReverseStatsError):', content)
        
        # We need to insert the import statement
        # Try to find a good place, like after the last typical import
        if 'from .exceptions import ReverseStatsError' not in content:
            # Let's just insert it at the very top, but after the module docstring
            # Or simpler: find first "import " or "from " and insert before it
            import_idx = content.find('\nimport ')
            import_from_idx = content.find('\nfrom ')
            
            insert_idx = -1
            if import_idx != -1 and import_from_idx != -1:
                insert_idx = min(import_idx, import_from_idx)
            elif import_idx != -1:
                insert_idx = import_idx
            elif import_from_idx != -1:
                insert_idx = import_from_idx
                
            if insert_idx != -1:
                # Add it right before the first import
                content = content[:insert_idx] + "\nfrom .exceptions import ReverseStatsError" + content[insert_idx:]
            else:
                # Fallback to top
                content = "from .exceptions import ReverseStatsError\n" + content
                
        with open(f_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Refactored {f_path}")
