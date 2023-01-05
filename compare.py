import argparse
import dill
import ast
import numpy as np
from train import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

class Transformer(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        if type(node.body[0]) == ast.Expr:
            del node.body[0]
        for elem in node.body:
            if type(elem) == ast.FunctionDef:
                if type(elem.body[0]) == ast.Expr:
                    del elem.body[0]
        return node

    def visit_FunctionDef(self, node):
        if type(node.body[0]) == ast.Expr:
            del node.body[0]
        return node

def my_open(text:str):
    """Open file with some errors.

    :param text: text of the opening file
    :return: text of the corrected file
    """
    try:
        ast.parse(text)
    except IndentationError as er:
        index = int(str(er).split('line ')[1].split()[0])
        text = text.split('\n')
        text.insert(index, '    """abc"""')
        text = '\n'.join(text)
    except SyntaxError as er:
        index = int(str(er).split('line ')[1][:-1])
        text = text.split('\n')
        line = text[index - 1]
        while 'in' in line:
            idx = line.index('in')
            if line[idx + 2] == '(':
                if line[idx - 4:idx - 1] == 'def':
                    line = line[:idx + 2] + 'i' + line[idx + 2:]
                else:
                    line = line[:idx + 2] + 't' + line[idx + 2:]
            elif '(in)' in line:
                ind = line.index('(in)')
                line = line[:ind] + '(ini)' + line[ind + 4:]
            elif 'in.' in line:
                ind = line.index('in.')
                line = line[:ind + 2] + 'i' + line[ind + 2:]
            elif 'in[' in line:
                ind = line.index('in[')
                line = line[:ind + 2] + 'i' + line[ind + 2:]
            elif 'in,' in line:
                ind = line.index('in,')
                line = line[:ind + 2] + 't' + line[ind + 2:]
            elif 'in=' in line:
                ind = line.index('in=')
                line = line[:ind + 2] + 't' + line[ind + 2:]
            elif 'in]' in line:
                ind = line.index('in]')
                line = line[:ind + 2] + 't' + line[ind + 2:]
            elif 'in)' in line:
                ind = line.index('in)')
                if 'def' in line:
                    line = line[:ind + 2] + 't' + line[ind + 2:]
                else:
                    line = line[:ind + 2] + 'i' + line[ind + 2:]
            elif 'in:' in line:
                ind = line.index('in:')
                if line[ind - 3:ind - 1] == '->':
                    line = line[:ind + 2] + 't' + line[ind + 2:]
                else:
                    line = line[:ind + 2] + 'i' + line[ind + 2:]
            if 'in =' in line:
                ind = line.index('in =')
                line = line[:ind] + 'ini' + line[ind + 2:]
            elif ' in' in line and ' ini' not in line and ' in ' not in line:
                ind = line.index(' in')
                line = line[:ind + 3] + 'i' + line[ind + 3:]
            else:
                break
        while 'self' in line:
            idx = line.index('self')
            if line[idx + 4] != ',' and line[idx + 4] != '.' and line[idx + 4] != ')':
                ind = idx + 4
                while line[ind] != ',' and line[ind] != '.' and line[ind] != ')':
                    ind += 1
                line = line[:idx + 4] + line[ind:]
            else:
                break
        while 'STR' in line:
            idx = line.index('STR')
            line = line[:idx] + 'str' + line[idx + 3:]
        while 'MIN' in line:
            idx = line.index('MIN')
            line = line[:idx] + 'min' + line[idx + 3:]
        while 'MAX' in line:
            idx = line.index('MAX')
            line = line[:idx] + 'max' + line[idx + 3:]
        while 'is' in line:
            if 'is(' in line:
                ind = line.index('is(')
                line = line[:ind] + 'isinstance' + line[ind + 2:]
            elif '(is)' in line:
                ind = line.index('(is)')
                line = line[:ind] + '(is_)' + line[ind + 4:]
            elif 'is,' in line:
                ind = line.index('is,')
                line = line[:ind + 2] + '_' + line[ind + 2:]
            elif 'is.' in line:
                ind = line.index('is.')
                line = line[:ind + 2] + '_' + line[ind + 2:]
            elif 'is:' in line:
                ind = line.index('is:')
                line = line[:ind + 2] + '_' + line[ind + 2:]
            elif ' is' in line and ' is_' not in line and ' is ' not in line:
                ind = line.index(' is')
                line = line[:ind + 3] + '_' + line[ind + 3:]
            else:
                break
        while 'for' in line:
            if '(for)' in line:
                ind = line.index('(for)')
                line = line[:ind] + '(fore)' + line[ind + 4:]
            elif ' for' in line and ' fore' not in line and ' for ' not in line:
                ind = line.index(' for')
                line = line[:ind + 3] + 'e' + line[ind + 3:]
            elif 'for,' in line:
                ind = line.index('for,')
                line = line[:ind + 2] + 'e' + line[ind + 2:]
            elif 'for.' in line:
                ind = line.index('for.')
                line = line[:ind + 2] + 'e' + line[ind + 2:]
            elif 'for:' in line:
                ind = line.index('for:')
                line = line[:ind + 2] + 'e' + line[ind + 2:]
            if 'for =' in line:
                ind = line.index('for =')
                line = line[:ind] + 'fore' + line[ind + 3:]
            else:
                break
        while 'def' in line:
            if '(def)' in line:
                ind = line.index('(def)')
                line = line[:ind] + '(defa)' + line[ind + 4:]
            elif ' def' in line and ' defa' not in line and ' def ' not in line:
                ind = line.index(' def')
                line = line[:ind + 3] + 'a' + line[ind + 3:]
            elif 'def,' in line:
                ind = line.index('def,')
                line = line[:ind + 2] + 'a' + line[ind + 2:]
            elif 'def.' in line:
                ind = line.index('def.')
                line = line[:ind + 2] + 'a' + line[ind + 2:]
            elif 'def:' in line:
                ind = line.index('def:')
                line = line[:ind + 2] + 'a' + line[ind + 2:]
            if 'def =' in line:
                ind = line.index('def =')
                line = line[:ind] + 'defa' + line[ind + 3:]
            else:
                break
        while 'return' in line:
            if '(return)' in line:
                ind = line.index('(return)')
                line = line[:ind] + '(returns)' + line[ind + 8:]
            elif ' return' in line and ' returns' not in line and ' return ' not in line:
                ind = line.index(' return')
                line = line[:ind] + ' returns' + line[ind + 7:]
            elif 'return,' in line:
                ind = line.index('return,')
                line = line[:ind] + 'returns' + line[ind + 6:]
            elif 'return.' in line:
                ind = line.index('return.')
                line = line[:ind] + 'returns' + line[ind + 6:]
            elif 'return:' in line:
                ind = line.index('return:')
                line = line[:ind] + 'returns' + line[ind + 6:]
            else:
                break
        while 'del' in line:
            if '(del)' in line:
                ind = line.index('(del)')
                line = line[:ind] + '(delt)' + line[ind + 4:]
            elif ' del' in line and ' delt' not in line and ' del ' not in line:
                ind = line.index(' del')
                line = line[:ind + 3] + 't' + line[ind + 3:]
            elif 'del,' in line:
                ind = line.index('del,')
                line = line[:ind + 2] + 't' + line[ind + 2:]
            elif 'del.' in line:
                ind = line.index('del.')
                line = line[:ind + 2] + 't' + line[ind + 2:]
            elif 'del:' in line:
                ind = line.index('del:')
                line = line[:ind + 2] + 't' + line[ind + 2:]
            if 'del =' in line:
                ind = line.index('del =')
                line = line[:ind] + 'delt' + line[ind + 3:]
            else:
                break

        text[index - 1] = line
        text = '\n'.join(text)
        text = my_open(text)
    return text
def load_file(filename:str):
    transformer = Transformer()
    with open(filename, "br") as f:
        text = f.read().decode()
        text = my_open(text)
        ast_tree = ast.parse(text)
        new_tree = transformer.visit(ast_tree)
    return ast.unparse(new_tree)
def levenshtein_distance(a: str, b: str) -> int:
    """Calculate The Levenshtein distance

    :param a: the first string
    :param b: the second string
    :return: them Levenshtein distance
    """
    m, n = len(a) + 1, len(b) + 1
    d0 = 0
    d1 = np.zeros(n, dtype=int)
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                d0 = 0
                d1[j] = 0
            elif i == 0:
                d1[j] = j
            elif j == 0:
                d0 = i
            else:
                k = 0 if a[i-1] == b[j-1] else 1
                d =  min(d0 + 1, d1[j] + 1, d1[j-1] + k)
                d1[j-1] = d0
                if j == n-1:
                    d1[j] = d
                d0 = d
    return d0

with open(args.input) as f:
    lines = [line.split() for line in f]
with open('model.pkl', 'rb') as f:
    lr = dill.load(f)

x = []
for line in lines:
    file1 = load_file(line[0])
    file2 = load_file(line[1])
    print(f'Считаем расстояние {lines.index(line) + 1} пары из {len(lines)}')
    x.append(levenshtein_distance(file1, file2) / max(len(file1), len(file2), 1))
x = np.array(x)
preds = np.around(lr.predict(x), decimals=3)

with open(args.output, 'w') as f:
    print(*preds, sep='\n', file=f)