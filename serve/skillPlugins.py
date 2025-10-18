import argparse, subprocess

skills = {
    "anime": "../skills/sol_anime",
    "study": "../skills/sol_studybuddy",
    "barista": "../skills/sol_barista"
}

parser = argparse.ArgumentParser()
parser.add_argument("skill", choices=skills.keys())
args = parser.parse_args()

subprocess.run(["python", "chat.py", "--adapter", skills[args.skill]])
