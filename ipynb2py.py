import json

def convert_ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r',encoding='utf-8') as f:
        notebook = json.load(f)

    with open(py_file, 'w',encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                f.write(''.join(cell['source']) + '\n\n')


if __name__ == '__main__':
    convert_ipynb_to_py("/root/wjh/gemma-2-9b-train.ipynb", "/root/wjh/train_gemma2.py")