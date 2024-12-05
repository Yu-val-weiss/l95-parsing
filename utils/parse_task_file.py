import re
from json import dump

with open("task/task_raw.txt", "r") as f:
    data = f.read()
    
section_pattern = section_pattern = re.compile(
    r'(\d+)\.\s+(.*?)\n'  # Section number and sentence
    r'((?:[^\n]+\n)+?)'  # PoS tags
    r'(\(S(?:.|\n)*?\.\))\n'  # Parse tree
    r'((?:\d+\t[^\n]+\n)+)'  # Dependency relations
)
sections = section_pattern.findall(data)

parsed_sections = {}


for section in sections:
    section_number = int(section[0])
    sent = section[1].strip()
    pos_tags = section[2].strip()
    parse_tree = section[3].strip()
    dep_rel = section[4].strip()
    
    parsed_sections[section_number] = {
        "sent": sent,
        "pos_tags": pos_tags,
        "parse_tree": parse_tree,
        "dependencies": dep_rel
    }
    
with open("task/task.json", "w") as tf:
    dump(parsed_sections, tf, indent=4)
    
with (open("task/sentences.txt", "w") as sf, open("task/pos_tags.txt", "w") as posf,
      open("task/parses.txt", "w") as parsef, open("task/dep_rel.txt", "w") as df):
    files = [sf, posf, parsef, df]
    for num in parsed_sections:
        for cat, f in zip(parsed_sections[num], files):
            f.write(parsed_sections[num][cat])
            f.write("\n\n")