"""SQL parsing + AST edit-script extraction for Approach 1 (SQLGlot based).

This is the ONLY module that touches SQLGlot.  `extract_pair(q, r)` returns a plain
dict ("record") that contains everything the models need:

    ok        True if both queries parsed and sqlglot.diff succeeded
    edits     list of (op, label_c, label_f, label2_c, label2_f, clause)
                op     : "ins" | "rem" | "upd" | "mov"   (Keep edits are dropped)
                label_c: coarse node label  (e.g. "Column", "GT", "Literal")
                label_f: fine node label    (e.g. "Identifier:order_dow", "Literal:10")
                label2_*: label of the target node for updates (same as label_* otherwise)
                clause : SQL clause the edited node lives in (select/from/join/where/...)
    q_c, q_f  coarse / fine labels of every node of the original query's AST
    r_c, r_f  same for the repaired query's AST
    tok       token-level distance (0..1) used by the documented fallback
"""
import difflib
import re

import sqlglot
from sqlglot import exp
from sqlglot.diff import Insert, Keep, Move, Remove, Update
from sqlglot.diff import diff as _sg_diff

DIALECT = "sqlite"   # the course databases are SQLite (backticks, "double quotes")
MAX_NODES = 2000     # safety guard: bigger trees use the fallback (keeps runtime bounded)

_CLAUSE_SPEC = [
    ("Where", "where"), ("Group", "group"), ("Having", "having"), ("Order", "order"),
    ("Limit", "limit"), ("Join", "join"), ("From", "from"), ("Distinct", "distinct"),
    ("Union", "setop"), ("Intersect", "setop"), ("Except", "setop"),
]
_CLAUSES = [(getattr(exp, n), c) for n, c in _CLAUSE_SPEC if hasattr(exp, n)]

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|'[^']*'|\"[^\"]*\"|`[^`]*`|<=|>=|<>|!=|\|\||[^\s\w]"
)


# ----------------------------------------------------------------------------- parsing
def parse_sql(sql):
    """Parse one SQL string (SQLite dialect). Raises on empty/invalid SQL."""
    s = (sql or "").strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    if not s:
        raise ValueError("empty query")
    trees = [t for t in sqlglot.parse(s, read=DIALECT) if t is not None]
    if not trees:
        raise ValueError("no statement")
    return trees[0]


def all_nodes(tree):
    return list(tree.find_all(exp.Expression))


# ----------------------------------------------------------------------------- labels
def node_labels(node):
    """(coarse, fine) label.  coarse = node type; fine adds the value of leaf-like nodes."""
    t = type(node).__name__
    v = getattr(node, "this", None)
    if isinstance(v, str):
        return t, f"{t}:{v.strip().lower()[:40]}"
    return t, t


def clause_of(node):
    """Name of the SQL clause that contains `node` (nearest enclosing clause)."""
    cur = node
    while cur is not None:
        for cls, name in _CLAUSES:
            if isinstance(cur, cls):
                return name
        parent = getattr(cur, "parent", None)
        if isinstance(cur, exp.Select) and parent is None:
            return "select"
        if isinstance(parent, exp.Select) and getattr(cur, "arg_key", None) == "expressions":
            return "select"
        cur = parent
    return "other"


# ----------------------------------------------------------------------------- edit script
def edit_script(tq, tr):
    """Edits (non-Keep) from SQLGlot's ChangeDistiller-based `sqlglot.diff`.

    Counting rules (one edit = one AST node):
      Insert  : a node of r with no counterpart in q
      Remove  : a node of q with no counterpart in r
      Update  : a matched pair of nodes whose content differs (substitution)
      Move    : a matched node that changed parent/position
    """
    out = []
    for e in _sg_diff(tq.copy(), tr.copy()):
        if isinstance(e, Keep):
            continue
        if isinstance(e, Insert):
            n1 = n2 = e.expression
            op = "ins"
        elif isinstance(e, Remove):
            n1 = n2 = e.expression
            op = "rem"
        elif isinstance(e, Update):
            n1, n2, op = e.source, e.target, "upd"
        elif isinstance(e, Move):
            n1, n2, op = e.source, e.target, "mov"
        else:
            continue
        c1, f1 = node_labels(n1)
        c2, f2 = node_labels(n2)
        out.append((op, c1, f1, c2, f2, clause_of(n1 if op in ("ins", "rem", "mov") else n2)))
    return out


# ----------------------------------------------------------------------------- fallback
def sql_tokens(sql):
    return [t.lower() for t in _TOKEN_RE.findall(sql or "")]


def token_distance(q, r):
    """1 - SequenceMatcher ratio over SQL tokens (0 = identical, 1 = nothing shared)."""
    a, b = sql_tokens(q), sql_tokens(r)
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# ----------------------------------------------------------------------------- main entry
def extract_pair(q, r):
    rec = {"ok": False, "edits": [], "q_c": [], "q_f": [], "r_c": [], "r_f": [],
           "tok": token_distance(q, r), "err": None}
    try:
        tq, tr = parse_sql(q), parse_sql(r)
        nq, nr = all_nodes(tq), all_nodes(tr)
        if len(nq) > MAX_NODES or len(nr) > MAX_NODES:
            raise ValueError("tree too large")
        edits = edit_script(tq, tr)
        lq = [node_labels(n) for n in nq]
        lr = [node_labels(n) for n in nr]
        rec.update(ok=True, edits=edits,
                   q_c=[a for a, _ in lq], q_f=[b for _, b in lq],
                   r_c=[a for a, _ in lr], r_f=[b for _, b in lr])
    except Exception as ex:  # parse error, diff error, recursion, ...
        rec["err"] = type(ex).__name__
    return rec
