with open("chat_history.txt", "r", encoding="utf-8") as f:
    text = f.read()

cleaned_lines = []
for line in text.splitlines():
    if ": " in line:
        new_line = line.split(": ", 1)[1]
        cleaned_lines.append(new_line)
    else:
        cleaned_lines.append(line)

# Filter out lines that are too short and empty lines
filtered_lines = [
    line for line in cleaned_lines if len(line) > 0 and line.strip() != ""
]

# add `0,"` in the beginning of each line and `"` at the end of each line
filtered_lines = [f'0,"{line}"' for line in filtered_lines]
result = "\n".join(filtered_lines)

with open("cleaned.txt", "w", encoding="utf-8") as f:
    f.write(result)
    print("Cleaned chat log saved to cleaned.txt")
