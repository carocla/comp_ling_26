completed                = False         # Change this flag to True when you've completed the assignment.
expected_completion_date = '01/17/2024'  # If your assignment is late, change this date to your expected completion date.
questions_or_comments    = ""            # Fill in this string with any questions or comments you have; leave empty if none.
extensions               = False         # Change this flag to True if you completed any extensions for this assignment.
extensions_description   = ""            # If you did any extensions, briefly explain what you did and where we should look for it.


def del_cost():
    return 1

def ins_cost():
    return 1

def sub_cost(c1, c2):
    if c1 == c2: 
        return 0
    else:
        return 2

def min_edit_distance(source, target, do_print_chart=False):
    """Compare `source` and `target` strings and return their edit distance with
    Levenshtein costs, according to the algorithm given in SLP Ch. 2, Figure 2.17.

    Parameters
    ----------
    source : str
        The source string.
    target : str
        The target string.

    Returns
    -------
    int
        The edit distance between the two strings.
    """
    n = len(source) + 1
    m = len(target) + 1
    D = [[0] * (m) for _ in range(n)]
    D[0][0] = 0
    for i in range(1, n):
        D[i][0] = D[i - 1][0] + del_cost()
    for j in range(1, m):
        D[0][j] = D[0][j - 1] + ins_cost()

    for i in range(1, n):
        for j in range(1, m):
            D[i][j] = min(
                D[i - 1][j] + del_cost(),
                D[i - 1][j - 1] + sub_cost(source[i-1], target[j-1]),
                D[i][j - 1] + ins_cost()
            )
    return D[n-1][m-1]
    # >>> END YOUR ANSWER
        
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 3:
        w1 = sys.argv[1]
        w2 = sys.argv[2]
    else:
        w1 = 'intention'
        w2 = 'execution'
    print('edit distance between', repr(w1), 'and', repr(w2), 'is', min_edit_distance(w1, w2))


