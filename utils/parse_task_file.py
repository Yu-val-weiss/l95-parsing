"""Run as a script to parse the raw task file."""

import re
from json import dump
from pathlib import Path

from depedit import DepEdit

from utils.conllu import convert_to_dep_rel, generate_conll, load_conll
from utils.task_data import dump_dep_rel

with Path("task_files/task_raw.txt").open() as f:
    data = f.read()

section_pattern = re.compile(
    r"(\d+)\.\s+(.*?)\n"  # Section number and sentence
    r"((?:[^\n]+\n)+?)"  # PoS tags
    r"(\(S(?:.|\n)*?\))\n"  # Parse tree
    r"((?:\d+\t[^\n]+\n)+)",  # Dependency relations
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
        "dependencies": dep_rel,
    }

with Path("task_files/task.json").open("w") as tf:
    dump(parsed_sections, tf, indent=4)

with (
    Path("task_files/sentences.txt").open("w") as sf,
    Path("task_files/pos_tags.txt").open("w") as posf,
    Path("task_files/parses.txt").open("w") as parsef,
    Path("task_files/dep_rel.txt").open("w") as df,
):
    files = [sf, posf, parsef, df]
    for section in parsed_sections.values():
        for cat, f in zip(section, files):
            f.write(section[cat])
            f.write("\n\n")

generate_conll()
dep = DepEdit("utils/stan2uni.ini")
with Path("task_files/task.conllu").open() as f:
    fixed = dep.run_depedit(infile=f.read())
with Path("task_files/task_fixed.conllu").open("w") as f:
    f.write(fixed)
fixed_df = load_conll("task_files/task_fixed.conllu")
dump_dep_rel(convert_to_dep_rel(fixed_df), "task_files/dep_rel_fixed.txt")
