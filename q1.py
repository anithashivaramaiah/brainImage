# Q1. Write a program that counts all right-leaning binary trees with N internal nodes. A binary tree is right-leaning if the size of its right branch is larger or equal than the size of its left branch.
def bin(n):
    if n == 0:
        yield 'o'
    else:
        for k in range(0, n):
            for l in bin(k):
                for r in bin(n - 1 - k):
                    yield l, r


class Tree:
    def __init__(self, Data):
        self.data = Data
        self.left = None
        self.right = None


def depth(tr):
    if tr:
        a = depth(tr.left)
        b = depth(tr.right)
        return max(a, b) + 1
    else:
        return 0


def prefixer(tr):
    if len(tr) > 1:
        return '*' + prefixer(tr[0]) + prefixer(tr[1])
    else:
        return tr


def converetr(a):
    if (a == ''):
        return ''
    if a[0] != '*':
        return Tree(a[0]), a[1:]
    else:
        p = Tree(a[0])
        p.left, q = converetr(a[1:])
        p.right, q = converetr(q)
        return p, q


def toBintree(bt):
    if bt is None: return
    if bt.left is None and bt.right is None:
        return bt.data
    else:
        return toBintree(bt.left), toBintree(bt.right)


if __name__ == "__main__":
    n = int(input("Enter n value: "))
    tres_expr = list(bin(n))
    count = 0
    for t in tres_expr:
        pf = prefixer(t)
        bt, _ = converetr(pf)
        if depth(bt.right) >= depth(bt.left):
            count += 1
    print("Right-leaning binary trees  = " + str(count))

# Output:
# Enter n value: 5
# Right-leaning binary trees  = 23
#
# Process finished with exit code 0