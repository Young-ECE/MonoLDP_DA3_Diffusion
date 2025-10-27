import re

def check_chinese_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chinese_comment_pattern = re.compile(r'#.*[\u4e00-\u9fff]+')
    chinese_comments = []

    for line_number, line in enumerate(lines, start=1):
        if chinese_comment_pattern.search(line):
            chinese_comments.append((line_number, line.strip()))

    return chinese_comments

file_path = 'trainer.py'  # 替换为你的文件路径
chinese_comments = check_chinese_comments(file_path)

if chinese_comments:
    print("Found Chinese comments in the following lines:")
    for line_number, comment in chinese_comments:
        print(f"Line {line_number}: {comment}")
else:
    print("No Chinese comments found.")