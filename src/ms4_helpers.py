"""Block helpers shared by the parts of the version-4 manuscript."""
B = []
FIG = "/home/claude/results/"
def title(t): B.append({"t": "title", "x": t})
def h1(t): B.append({"t": "h1", "x": t})
def h2(t): B.append({"t": "h2", "x": t})
def p(t): B.append({"t": "p", "x": t})
def eq(t, n): B.append({"t": "eq", "x": t, "n": n})
def bullets(xs): B.append({"t": "bullets", "x": xs})
def fig(path, cap, w=6.3): B.append({"t": "fig", "path": FIG + path, "cap": cap, "w": w})
def pb(): B.append({"t": "pb"})
def meta(t): B.append({"t": "meta", "x": t})
def tab(key): B.append({"t": key})
