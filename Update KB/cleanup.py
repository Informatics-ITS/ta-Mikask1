import re

def clean_content(content):
    if not content:
        return content
        
    # Remove excessive newlines
    content = re.sub(r'\n{2,}', '\n', content)
    
    # Remove excessive whitespaces
    content = re.sub(r' {2,}', ' ', content)
    content = re.sub(r'\t{2,}', '\t', content)
    content = re.sub(r' +$', '', content, flags=re.MULTILINE)
    
    # Remove ' - number' patterns
    content = re.sub(r' - \d+', '', content)
    
    # Remove lines containing "TAMBAHAN LEMBARAN NEGARA REPUBLIK INDONESIA"
    content = re.sub(r'TAMBAHAN LEMBARAN NEGARA REPUBLIK INDONESIA.*$\n?', '', content, flags=re.MULTILINE|re.IGNORECASE)
    
    # Remove '# - ' patterns
    content = re.sub(r'^# -$\n?', '', content, flags=re.MULTILINE)
    
    # Replace '## -N-' patterns with just the number
    content = re.sub(r'## -(\d+)-', r'\1', content, flags=re.MULTILINE)
    
    # Remove '#number' patterns
    content = re.sub(r'^#\d+\s*$', '', content, flags=re.MULTILINE)
    
    # Remove lines containing only numbers
    content = re.sub(r'^\s*\d+\s*$\n?', '', content, flags=re.MULTILINE)
    
    # Remove '-----\nPRESIDEN\n[LINE]' pattern
    content = re.sub(r'-{3,}\nPRESIDEN\n.*?\n', '---\n', content, flags=re.MULTILINE | re.DOTALL)
    
    # Add line breaks after horizontal rules (---)
    content = re.sub(r'^-{3,}$', '---', content, flags=re.MULTILINE)
    content = re.sub(r'---(?!\n)', '---\n', content)
    
    # Replace 'Cukup [jelas.]' with 'Cukup jelas'
    content = re.sub(r'Cukup \[jelas\.\]', 'Cukup jelas', content)
    
    # Remove everything from '### II. PASAL DEMI PASAL' to the end of the file
    content = re.sub(r'### II\. PASAL DEMI PASAL.*$', '', content, flags=re.DOTALL)
    
    # Remove standalone opening parentheses without closing parentheses
    content = re.sub(r'\((?![^\(]*\))', '', content)
    
    # Remove leading spaces from lines that start with a space
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.startswith(' '):
            cleaned_lines.append(line[1:])
        else:
            cleaned_lines.append(line)
    content = '\n'.join(cleaned_lines)
    
    # Remove all ellipsis patterns
    content = re.sub(r'\.{3}', '', content)
    content = re.sub(r'\. \. \.', '', content)
    content = re.sub(r'…', '', content)
    
    # Remove all triple backticks
    content = re.sub(r'```', '', content)
    
    # Replace parenthesized numbers like (4) with just the number
    content = re.sub(r'\((\d+)\)', r'\1 ', content)
    
    # Replace semicolons with periods
    content = re.sub(r';', '.', content)
    
    # Remove unmatched double quotes
    content = re.sub(r'(?<!")"(?!")|"(?=\n|$)', '', content)
    
    # Remove links
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
    
    # Replace [content] with content
    content = re.sub(r'\[([^\]]+)\]', r'\1', content)
    
    # Format currency references properly
    content = re.sub(r'[rR][pP]\.?\s*(\d[\d\.,]*)', r'Rp\1', content)
    content = re.sub(r'[rR][pP][a-zA-Z]+(\d[\d\.,]*)', r'Rp\1', content)
    content = re.sub(r'([rR][pP][\d\.,]+)[a-zA-Z]+', r'\1', content)
    content = re.sub(r'([rR][pP][\d\.,]+)\s+([lI\d\.,]+)', r'\1\2', content)
    content = re.sub(r'([rR][pP][\d\.,]+)[lI]([lI\d\.,]+)', r'\1\2', content)
    content = re.sub(r'([rR][pP][\d\.,]+)O([lI\d\.,]+)', r'\1\2', content)
    
    return content
