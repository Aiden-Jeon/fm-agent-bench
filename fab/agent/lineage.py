"""Agent 실행의 최종 제출 계보 (C1 준비, docs/tabarena.md 의 "C1 준비").

    uv run --frozen fab ta agent lineage <key 또는 실행 폴더>            # 사람이 읽는 나무
    uv run --frozen fab ta agent lineage <key> --json --out lineage.json  # 기계가 읽는 JSON

회수한 실행 폴더 (fab/agent/runner.py 의 회수물: meta.json · 하네스 기록 · workspace.tgz · submission.csv) 만 읽음.
key 는 AGENT_RUNS (기본은 메인 체크아웃의 results/_mlebench/agents/, FAB_OUTPUTS 로 바꿈) 아래에서 찾음.

하는 일 (정해진 방식으로 찾을 수 있는 것만):
  1. 하네스 기록을 순서대로 "실행" 목록으로 바꿈
     - Codex (`codex.jsonl`): command_execution 의 명령 (bash -lc 안쪽). Claude Code (`claude-code.jsonl`): Bash 명령 +
       Write·Edit 도구. 명령 안의 `python x.py` · `python - <<'PY' … PY` (인라인) · `python -c` · `bash x.sh` · `cat > x.py <<EOF` ·
       `cp`/`mv` · `sed -i` · `> 파일`. 실패한 명령 (exit ≠ 0, Claude Code is_error) 은 쓰기로 안 셈
     - AIDE (`workspace/logs/exp/journal.json`): 노드마다 코드 하나 = 실행 하나 (step 순). 최종 = best_solution 노드
  2. 제출 파일 (meta.json 의 submission_from) 을 마지막으로 쓴 실행을 찾고, 그 코드가 읽는 파일을 만든 실행을 거꾸로 따라감
     - 코드의 파일 경로는 AST 로: 글자·f-string ({…} 는 `*`)·Path `/`·os.path.join·변수 대입. 부른 함수로 역할을 정함
       (np.load·read_csv·open(r) = 읽기, np.save·to_csv·open(w)·joblib.dump = 쓰기, shutil.copy·Path.replace = 복사).
       함수 인자로 넘어가 역할을 모르는 경로는 "언급" (읽기로 봄, 쓰기로는 안 셈)
     - 파일을 만든 실행 = 그 파일을 쓴, 소비자보다 앞선 실행: 정확한 경로로 쓴 것이 있으면 그 마지막, 없으면 경로가 가장
       구체적인 (`*` 밖 글자가 많은) 것의 마지막. 가장 구체적인 쓰기가 서로 다른 코드 둘 이상이면 "여럿" 으로 두고 고르지 않음.
       쓰기가 없으면 "못 찾음". 같은 코드가 쓰는 경로를 첫 쓰기보다 앞에서 읽으면 읽고 고쳐 다시 쓰기 (rewrites) → 앞선 쓰기로
     - exec(·compile(·runpy 로 다른 .py 를 읽어 돌리면 그 파일의 쓰기도 이 실행의 쓰기 (글을 잘라 돌리면 안 셈, 글자를 고쳤으면
       exec_text_modified). 명령 앞의 환경 변수 (VERSION=6 등) 는 실행마다 env
     - 실행 시점의 코드: 기록에 전체 내용이 남은 (heredoc·Write, Edit 는 적용) 판이 있으면 그것, 아니면 작업 디렉터리 최종본.
       최종본을 썼는데 실행 뒤에 그 파일을 고친 명령이 있으면 changed_after_run 으로 표시 (실행한 판과 다를 수 있음)
  3. 코드마다 import (최상위 모듈) 와 로컬 모듈 (작업 디렉터리에 <모듈>.py 가 있는 것, 그 모듈의 파일 경로도 같이 따라감)
작업 디렉터리 코드는 pip 가 깐 파일 (`*.dist-info/RECORD`, fab/agent 의 installed_files) 을 뺌 (사후 검사와 같은 규칙).

하지 않는 일: 피처의 뜻, 타깃 의존·train+test fit (inductive) 여부, 섞기 무게의 해석 → 카드에서 사람·LLM 이 읽고 채움
(C1 규칙 1 "LLM 변환 + 사람 검토"). 쓰는 경로를 셸 변수·argparse·명령 출력으로만 알 수 있는 경우는 못 찾음으로 남음.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import json
import posixpath
import re
import shlex
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

WORK = "/tmp/mleb_work"  # runner.py 의 WORK (하네스 작업 디렉터리 = 모든 하네스의 cwd)
HOME = "/tmp/mleb_home"  # runner.py 의 HOME
DATA = f"{HOME}/data"
AIDE_EXP = f"{WORK}/workspaces/exp"  # AIDE 의 노드 코드가 도는 곳 (./input = 데이터, ./working, ./submission)
DATA_NAMES = {"train.csv", "test.csv", "sample_submission.csv", "description.md"}
# 계보로 따라갈 파일 (예측·OOF·피처 배열·모델). 로그·글 (.log·.txt·.md) 은 안 따라감
FILE_EXT = {".npy", ".npz", ".csv", ".parquet", ".pq", ".pkl", ".pickle", ".joblib", ".feather", ".json", ".pt", ".pth",
            ".h5", ".hdf5", ".cbm", ".tsv", ".gz", ".bin"}
READ_FUNCS = {"load", "loadtxt", "genfromtxt", "read_csv", "read_parquet", "read_feather", "read_pickle", "read_json",
              "read_table", "load_npz", "fromfile", "imread", "read_hdf", "memmap", "open_memmap"}
WRITE_FUNCS = {"save", "savez", "savez_compressed", "savetxt", "to_csv", "to_parquet", "to_feather", "to_pickle",
               "to_json", "save_npz", "tofile", "savefig", "save_model", "dump", "to_hdf"}
RECEIVER_READ = {"read_text", "read_bytes"}
RECEIVER_WRITE = {"write_text", "write_bytes"}
COPY_FUNCS = {"copy", "copy2", "copyfile", "move", "rename", "replace"}  # shutil·os 의 (src, dst) 또는 Path.replace(dst)
SHUTIL_NAMES = {"copy", "copy2", "copyfile", "move"}  # from shutil import copy 꼴 (str.replace 와 헷갈리지 않게 replace·rename 은 뺌)
PATH_KEEP = {"resolve", "absolute", "expanduser", "as_posix"}  # 경로를 그대로 두는 메서드
WRAPPERS = {"nohup", "time", "env", "nice", "setsid", "exec", "stdbuf", "ionice", "taskset", "unbuffer"}
PYTHON = re.compile(r"^(?:.*/)?python(?:\d+(?:\.\d+)?)?$")
HEREDOC_OP = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
SHELL_OPS = {";", "&&", "||", "|", "&", "(", ")", "|&"}


# ---------- 코드 안의 파일 경로 ----------

@dataclass
class Ref:
    pattern: str  # 경로 (모르는 부분은 *), 실행 cwd 기준 상대 경로일 수 있음
    role: str  # read · write · mention
    line: int = 0
    col: int = 0  # 같은 줄 안 순서 (한 줄에 ; 로 이어 쓴 인라인 코드)


def _join(*parts: str) -> str:
    out = ""
    for p in parts:
        out = p if (not out or p.startswith("/")) else (out.rstrip("/") + "/" + p)
    return out


class _Paths(ast.NodeVisitor):
    """코드 하나의 경로 글자 → Ref. 변수 대입 (이름 = 경로 식) 은 코드 순서대로 기억 (함수 안팎 구분 없음, 근사)."""

    def __init__(self):
        self.env: dict[str, str] = {}
        self.loops: dict[str, list[str]] = {}  # for x in ['a', 'b'] (글자 목록) 의 x → 값들
        self.refs: list[Ref] = []
        self.used: set[int] = set()  # 역할을 정한 식 (id) → 언급으로 다시 안 셈

    # 경로 식 → 글 (모르면 None)
    def pat(self, n) -> str | None:
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if isinstance(n, ast.JoinedStr):
            out = []
            for v in n.values:
                if isinstance(v, ast.Constant):
                    out.append(str(v.value))
                else:
                    p = self.pat(v.value) if isinstance(v, ast.FormattedValue) else None
                    out.append(p if p is not None else "*")
            return re.sub(r"\*+", "*", "".join(out))
        if isinstance(n, ast.Name):
            if n.id in self.loops:
                return f"\x00{n.id}\x00"  # _expand 가 값마다 펼침
            return self.env.get(n.id)
        if isinstance(n, ast.BinOp):
            left, right = self.pat(n.left), self.pat(n.right)
            if isinstance(n.op, ast.Div) and (left is not None or right is not None):
                return _join(left if left is not None else "*", right if right is not None else "*")
            if isinstance(n.op, ast.Add) and (left is not None or right is not None):
                return re.sub(r"\*+", "*", (left if left is not None else "*") + (right if right is not None else "*"))
            if isinstance(n.op, ast.Mod) and left is not None:
                return re.sub(r"%[-#0 +]*\d*(?:\.\d+)?[sdifgrx]", "*", left)
            return None
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in ("Path", "PurePath", "PosixPath", "str") and len(n.args) == 1:
                return self.pat(n.args[0])
            if name == "join" and isinstance(f, ast.Attribute) and self._is_os_path(f.value) and n.args:
                ps = [self.pat(a) for a in n.args]
                return _join(*[p if p is not None else "*" for p in ps]) if any(p is not None for p in ps) else None
            if isinstance(f, ast.Attribute):
                recv = self.pat(f.value)
                if name == "format" and recv is not None:
                    return re.sub(r"\{[^}]*\}", "*", recv)
                if recv is None:
                    return None
                if name in PATH_KEEP:
                    return recv
                if name == "with_suffix" and n.args and (s := self.pat(n.args[0])) is not None:
                    return posixpath.splitext(recv)[0] + s
                if name == "with_name" and n.args and (s := self.pat(n.args[0])) is not None:
                    return _join(posixpath.dirname(recv), s) if posixpath.dirname(recv) else s
                if name == "joinpath" and n.args:
                    return _join(recv, *[self.pat(a) or "*" for a in n.args])
        return None

    @staticmethod
    def _is_os_path(n) -> bool:
        return (isinstance(n, ast.Attribute) and n.attr == "path") or (isinstance(n, ast.Name) and n.id in ("path", "osp"))

    def _expand(self, p: str) -> list[str]:
        """반복 변수 자리 (\x00x\x00) → 값마다 하나 (여럿이면 곱). 모르는 자리는 *."""
        names = re.findall(r"\x00(\w+)\x00", p)
        outs = [p]
        for nm in dict.fromkeys(names):
            vals = self.loops.get(nm) or ["*"]
            outs = [o.replace(f"\x00{nm}\x00", v) for o in outs for v in vals][:500]
        return [re.sub(r"\*+", "*", o) for o in outs]

    def _add(self, node, role):
        p = self.pat(node)
        if p is not None:
            for q in self._expand(p):
                self.refs.append(Ref(q, role, getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
            for sub in ast.walk(node):
                self.used.add(id(sub))

    def visit_For(self, n):
        vals = None
        if isinstance(n.target, ast.Name) and isinstance(n.iter, (ast.List, ast.Tuple)) and n.iter.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, (str, int)) for e in n.iter.elts):
            vals = [str(e.value) for e in n.iter.elts]
        if vals is None:
            self.generic_visit(n)
            return
        old = self.loops.get(n.target.id)
        self.loops[n.target.id] = vals
        self.generic_visit(n)
        if old is None:
            self.loops.pop(n.target.id, None)
        else:
            self.loops[n.target.id] = old

    def visit_Assign(self, n):
        self.generic_visit(n)
        if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and (p := self.pat(n.value)) is not None:
            self.env[n.targets[0].id] = p

    def visit_AnnAssign(self, n):
        self.generic_visit(n)
        if isinstance(n.target, ast.Name) and n.value is not None and (p := self.pat(n.value)) is not None:
            self.env[n.target.id] = p

    def visit_Call(self, n):
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
        args = list(n.args) + [k.value for k in n.keywords if k.arg in ("path", "file", "fname", "filepath_or_buffer",
                                                                        "path_or_buf", "f", "filename")]
        if name == "open" and args:
            mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
            m = mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else "r"
            if isinstance(f, ast.Attribute) and self.pat(f.value) is not None:  # Path(...).open(mode)
                m = n.args[0].value if n.args and isinstance(n.args[0], ast.Constant) else "r"
                self._add(f.value, "write" if re.search(r"[wax]", str(m)) else "read")
            else:
                self._add(args[0], "write" if re.search(r"[wax]", m) else "read")
        elif name == "replace" and isinstance(f, ast.Attribute) and len(n.args) == 2 and not (
                isinstance(f.value, ast.Name) and f.value.id in ("os", "shutil")):
            for a in n.args:  # 문자열 치환 s.replace('a.npz', 'b.npz') 의 글자는 파일 경로가 아님 (스크립트를 고치는 코드)
                for sub in ast.walk(a):
                    self.used.add(id(sub))
        elif name == "run_path" and args:  # runpy.run_path('x.py')
            self._add(args[0], "read")
        elif name in RECEIVER_READ and isinstance(f, ast.Attribute):
            self._add(f.value, "read")
        elif name in RECEIVER_WRITE and isinstance(f, ast.Attribute):
            self._add(f.value, "write")
        elif name in COPY_FUNCS and isinstance(f, ast.Attribute) and len(n.args) == 1 and self.pat(f.value) is not None \
                and self.pat(n.args[0]) is not None:  # Path.replace(dst) · Path.rename(dst)
            self._add(f.value, "read")
            self._add(n.args[0], "write")
        elif len(n.args) >= 2 and self.pat(n.args[0]) is not None and (
                (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("shutil", "os")
                 and name in COPY_FUNCS) or (isinstance(f, ast.Name) and name in SHUTIL_NAMES)):  # shutil.copy(src, dst)
            self._add(n.args[0], "read")
            self._add(n.args[1], "write")
        elif name in READ_FUNCS and args:
            self._add(args[0], "read")
        elif name in WRITE_FUNCS:
            # np.save(path, arr) · df.to_csv(path) 는 첫 인자, joblib.dump(obj, path) · torch.save(obj, path) 는 둘째
            cands = [a for a in args[:2] if (p := self.pat(a)) is not None and _is_file(p)]
            if not cands and name.startswith(("to_", "save")) and args and self.pat(args[0]) is not None:
                cands = args[:1]  # df.to_csv(out.with_suffix('.tmp')) 처럼 확장자가 FILE_EXT 밖이어도 첫 인자
            for a in cands[:1]:
                self._add(a, "write")
        self.generic_visit(n)

    def _mention(self, p: str, n):
        for q in self._expand(p):
            self.refs.append(Ref(q, "mention", getattr(n, "lineno", 0)))

    def visit_Constant(self, n):
        if isinstance(n.value, str) and id(n) not in self.used and _is_file(n.value) and not re.search(r"\s", n.value):
            self.refs.append(Ref(n.value, "mention", getattr(n, "lineno", 0)))

    def visit_JoinedStr(self, n):
        if id(n) not in self.used and (p := self.pat(n)) is not None and _is_file(p):
            self._mention(p, n)
            for sub in ast.walk(n):
                self.used.add(id(sub))
        self.generic_visit(n)

    def visit_BinOp(self, n):
        if id(n) not in self.used and isinstance(n.op, (ast.Div, ast.Add, ast.Mod)) and (p := self.pat(n)) is not None \
                and _is_file(p):
            self._mention(p, n)
            for sub in ast.walk(n):
                self.used.add(id(sub))
        self.generic_visit(n)


def _is_file(p: str) -> bool:
    ext = posixpath.splitext(p.rstrip("/"))[1].lower()
    return ext in FILE_EXT and len(p) < 400


def code_refs(code: str) -> list[Ref] | None:
    """코드의 파일 경로 (읽기·쓰기·언급). 파이썬으로 못 읽으면 None."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    v = _Paths()
    # 한 번 돌아 변수를 모은 뒤 다시 (위에서 쓰고 아래에서 정의한 이름도 풀리게)
    v.visit(tree)
    env = v.env
    v = _Paths()
    v.env = dict(env)
    v.visit(tree)
    roled = {r.pattern for r in v.refs if r.role != "mention"}
    seen, out = set(), []
    for r in v.refs:  # 같은 (경로, 역할) 은 한 번. 읽기·쓰기로 잡힌 경로의 언급 (대입한 글자) 은 뺌
        if r.role == "mention" and r.pattern in roled:
            continue
        if (r.pattern, r.role) not in seen:
            seen.add((r.pattern, r.role))
            out.append(r)
    return out


def code_imports(code: str) -> list[str]:
    """import 하는 최상위 모듈 (상대 import 는 '.<모듈>')."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(("." * n.level) + (n.module or "").split(".")[0] if n.level else (n.module or "").split(".")[0])
    return sorted(m for m in mods if m)


# ---------- 하네스 기록 → 실행 ----------

@dataclass
class Run:
    """코드 하나를 한 번 돌림 (스크립트 · 인라인 · AIDE 노드) 또는 셸 동작 (복사·리다이렉트)."""
    order: tuple  # (기록 순서, 명령 안 순서)
    kind: str  # script · inline · aide_node · copy · shell
    command: str  # 하네스 기록의 명령 (줄임 없이)
    ok: bool | None
    cwd: str
    path: str | None = None  # script: 절대 경로, copy: 대상
    src: str | None = None  # copy 의 원본
    code: str | None = None
    version: str = ""  # log · workspace · journal · missing
    changed_after_run: bool = False
    writes: list[str] = field(default_factory=list)  # 셸 리다이렉트·복사 대상 (절대 경로)
    node_id: str | None = None  # AIDE
    env: dict = field(default_factory=dict)  # 명령 앞의 환경 변수 대입 (VERSION=6 python x.py: 같은 코드가 다른 판으로 돎)


@dataclass
class FileEvent:
    order: tuple
    path: str
    content: str | None  # None = 내용 모름 (sed -i · 인라인 코드의 write_text 등)
    kind: str = ""
    edit: tuple | None = None  # Claude Code Edit: (old, new, replace_all)


def _abs(p: str, cwd: str) -> str:
    if p.startswith("~"):
        p = "/root" + p[1:]
    return posixpath.normpath(p if p.startswith("/") or p.startswith("*") else posixpath.join(cwd, p))


def split_heredocs(text: str) -> tuple[str, dict[int, str]]:
    """명령 글 → (heredoc 본문을 뺀 글, {본문이 붙은 줄 번호: 본문}). 한 줄에 heredoc 이 여럿이면 첫 것만 그 줄에 붙임."""
    lines, out, bodies, i = text.split("\n"), [], {}, 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        ops = list(HEREDOC_OP.finditer(line))
        i += 1
        for k, m in enumerate(ops):
            tag, body = m.group(2), []
            while i < len(lines) and lines[i].strip() != tag:
                body.append(lines[i])
                i += 1
            i += 1  # 끝 표시 줄
            if k == 0:
                bodies[len(out) - 1] = "\n".join(body)
    return "\n".join(out), bodies


def _tokens(line: str) -> list[str]:
    try:
        lex = shlex.shlex(line, posix=True, punctuation_chars=";&|()<>")
        lex.whitespace_split = True
        lex.commenters = ""
        return list(lex)
    except ValueError:
        return line.split()


def _simple_commands(tokens: list[str]) -> list[list[str]]:
    out, cur = [], []
    for t in tokens:
        if t in SHELL_OPS:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def _strip_redirects(words: list[str]) -> tuple[list[str], list[str], bool]:
    """낱말 → (리다이렉트를 뺀 낱말, > · >> 대상 파일, < 또는 << 가 있는지)."""
    out, targets, stdin, i = [], [], False, 0
    while i < len(words):
        w = words[i]
        if re.fullmatch(r"\d*(?:>>?|&>|>&)", w) or (w.isdigit() and i + 1 < len(words) and words[i + 1].startswith(">")):
            if w.isdigit():
                i += 1
                w = words[i]
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if nxt and not nxt.startswith("&") and not nxt.isdigit():
                targets.append(nxt)
            elif nxt == "&" and i + 2 < len(words):  # 2>&1 이 '>' '&' '1' 로 나뉜 경우
                i += 1
            i += 2
            continue
        if w in ("<", "<<", "<<-", "<<<"):
            stdin = True
            i += 2
            continue
        out.append(w)
        i += 1
    return out, targets, stdin


def _unwrap(words: list[str]) -> list[str]:
    """환경 변수 대입·감싸는 명령 (nohup·timeout N·env 등) 을 뗌."""
    i = 0
    while i < len(words):
        w = words[i]
        if re.fullmatch(r"[A-Za-z_]\w*=.*", w):
            i += 1
        elif w in WRAPPERS:
            i += 1  # 옵션과 그 숫자 값 (nice -n 5 · taskset 0x3 · stdbuf -oL)
            while i < len(words) and (words[i].startswith("-") or re.fullmatch(r"\d+|0x[0-9a-fA-F]+", words[i])):
                i += 1
        elif w == "timeout":
            i += 1
            while i < len(words) and (words[i].startswith("-") or re.fullmatch(r"\d+[smhd]?", words[i])):
                i += 1
        else:
            break
    return words[i:]


def unwrap_bash(cmd: str) -> str:
    """Codex 의 `/bin/bash -lc "<명령>"` → 안쪽 명령."""
    try:
        w = shlex.split(cmd)
    except ValueError:
        return cmd
    if len(w) >= 3 and w[0].rsplit("/", 1)[-1] in ("bash", "sh", "zsh") and re.fullmatch(r"-\w*c\w*", w[1]):
        return w[2]
    return cmd


PARTIAL = re.compile(r"\.split\(|\.partition\(|\.splitlines\(|\[\s*-?\d*\s*:|\.index\(|\.find\(")  # 읽은 코드 글을 자름


FOR_LOOP = re.compile(r"\bfor\s+([A-Za-z_]\w*)\s+in\s+([^;\n]*?)\s*(?:;|\n)\s*do\b(.*?)(?:;|\n)?\s*\bdone\b", re.S)


def expand_loops(text: str) -> str:
    """`for x in a b c; do …; done` → 몸을 x 마다 한 번씩 (값이 글자 목록일 때만: $( ) · * · $ 가 있으면 그대로)."""
    for _ in range(3):  # 겹친 반복
        def one(m):
            var, items, body = m.group(1), m.group(2).split(), m.group(3)
            if not items or any(re.search(r"[$`*?(){}]", w) for w in items):
                return m.group(0)
            items = [w.strip("'\"") for w in items]
            return "\n".join(re.sub(rf"\$\{{{var}\}}|\${var}\b", w, body).strip().rstrip("&").strip()
                             + (" &" if body.strip().endswith("&") else "") for w in items)
        new = FOR_LOOP.sub(one, text)
        if new == text:
            break
        text = new
    return text


def parse_shell(text: str, cwd: str, idx: int, ok: bool | None, read_file=None, depth: int = 0):
    """셸 명령 하나 → (실행 목록, 파일 사건 목록, 끝난 뒤 cwd). read_file(절대 경로) → 셸 스크립트 내용 (bash x.sh 를 풀 때).
    for 반복은 펼침 (expand_loops). heredoc 본문은 펼치기 전에 떼어 둠 (본문 안의 for 는 코드라서)."""
    body_text, bodies = split_heredocs(text)
    if FOR_LOOP.search(body_text):
        # 펼치면 줄 번호가 바뀌므로 heredoc 이 붙은 줄은 표시로 바꿔 두고 펼친 뒤 다시 찾음
        marked = "\n".join(f"{line} #@HD{k}" if k in bodies else line for k, line in enumerate(body_text.split("\n")))
        expanded = expand_loops(marked)
        new_bodies, lines = {}, expanded.split("\n")
        for k, line in enumerate(lines):
            if (m := re.search(r" #@HD(\d+)$", line)):
                new_bodies[k] = bodies[int(m.group(1))]
                lines[k] = line[:m.start()]
        body_text, bodies = "\n".join(lines), new_bodies
    runs, events, sub = [], [], 0
    for ln, line in enumerate(body_text.replace("\\\n", " ").split("\n")):
        heredoc = bodies.get(ln)
        for words in _simple_commands(_tokens(line)):
            sub += 1
            order = (idx, sub)
            words, targets, stdin = _strip_redirects(words)
            env = dict(w.split("=", 1) for w in words if re.fullmatch(r"[A-Za-z_]\w*=.*", w)
                       and not w.startswith(("OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_", "CUDA_VISIBLE", "PYTHONUNBUFFERED")))
            words = _unwrap(words)
            if not words:
                continue
            exe = words[0]
            targets_abs = [_abs(t, cwd) for t in targets if t not in ("/dev/null",)]
            if exe == "cd" and len(words) > 1:
                cwd = _abs(words[1], cwd)
                continue
            if PYTHON.match(exe):
                rest, code, script, i = words[1:], None, None, 0
                while i < len(rest):
                    w = rest[i]
                    if w in ("-W", "-X", "-Q"):
                        i += 2
                        continue
                    if w == "-c" and i + 1 < len(rest):
                        code = rest[i + 1]
                        break
                    if w == "-m":
                        script = None
                        code = ""
                        break
                    if w == "-":
                        code = heredoc if heredoc is not None else None
                        break
                    if w.startswith("-"):
                        i += 1
                        continue
                    script = w
                    break
                else:
                    if stdin and heredoc is not None:  # python <<'PY'
                        code = heredoc
                if script:
                    runs.append(Run(order, "script", text, ok, cwd, path=_abs(script, cwd), writes=targets_abs, env=env))
                elif code:
                    runs.append(Run(order, "inline", text, ok, cwd, code=code, version="log", writes=targets_abs, env=env))
                continue
            if exe.endswith(".py") and exe.startswith(("./", "/")):
                runs.append(Run(order, "script", text, ok, cwd, path=_abs(exe, cwd), writes=targets_abs))
                continue
            if exe in ("bash", "sh") and len(words) > 1 and words[1].endswith(".sh") and read_file and depth < 2:
                path = _abs(words[1], cwd)
                if (sh := read_file(path)) is not None:
                    r2, e2, _ = parse_shell(sh, cwd, idx, ok, read_file, depth + 1)
                    for r in r2:
                        r.order = (idx, sub) + r.order[1:]
                    runs += r2
                    events += e2
                continue
            if exe in ("cat", "tee") and heredoc is not None:
                for t in (targets_abs if exe == "cat" else [_abs(w, cwd) for w in words[1:] if not w.startswith("-")]):
                    events.append(FileEvent(order, t, heredoc, "heredoc"))
                continue
            if exe in ("cp", "mv", "install", "rsync") and len(words) >= 3:
                args = [w for w in words[1:] if not w.startswith("-")]
                if len(args) >= 2:
                    dst = _abs(args[-1], cwd)
                    for s in args[:-1]:
                        src = _abs(s, cwd)
                        to = dst if len(args) == 2 and _is_file(dst) or posixpath.basename(dst) == posixpath.basename(src) \
                            else posixpath.join(dst, posixpath.basename(src))
                        runs.append(Run(order, "copy", text, ok, cwd, path=to, src=src))
                        events.append(FileEvent(order, to, None, "copy"))
                continue
            if exe == "sed" and any(w == "-i" or w.startswith("-i") for w in words[1:]):
                args, i, script_given = [], 1, False
                while i < len(words):
                    w = words[i]
                    if w in ("-e", "-f"):
                        script_given = True
                        i += 2
                        continue
                    if not w.startswith("-"):
                        args.append(w)
                    i += 1
                files = args if script_given else args[1:]
                for f in files:
                    events.append(FileEvent(order, _abs(f, cwd), None, "sed -i"))
                continue
            if targets_abs:  # 그 밖의 명령의 > 파일 (echo · head · cat a > b 등)
                runs.append(Run(order, "shell", text, ok, cwd, writes=targets_abs))
                for t in targets_abs:
                    events.append(FileEvent(order, t, None, "redirect"))
    # 인라인 코드가 .py 파일을 고치면 (Path(x).write_text(...)) 내용 모름으로 표시
    for r in runs:
        if r.kind == "inline" and (refs := code_refs(r.code or "")):
            for ref in refs:
                if ref.role == "write" and ref.pattern.endswith(".py") and "*" not in ref.pattern:
                    events.append(FileEvent(r.order, _abs(ref.pattern, r.cwd), None, "inline write"))
    return runs, events, cwd


# ---------- 실행 폴더 ----------

class Workspace:
    """workspace.tgz (작업 디렉터리 = WORK, home_submission = HOME/submission). pip 가 깐 파일은 뺌."""

    def __init__(self, tgz: Path | None):
        from fab.agent import installed_files

        self.members: dict[str, tarfile.TarInfo] = {}
        self.tar = tarfile.open(tgz) if tgz and tgz.exists() else None
        if self.tar:
            skip = installed_files(self.tar)
            for m in self.tar.getmembers():
                name = posixpath.normpath(m.name)
                if m.isfile() and name not in skip:
                    self.members[self.abs(name)] = m

    @staticmethod
    def abs(name: str) -> str:
        if name == "workspace" or name.startswith("workspace/"):
            return WORK + name[len("workspace"):]
        if name.startswith("home_submission"):
            return f"{HOME}/submission" + name[len("home_submission"):]
        return "/?/" + name

    def read(self, path: str) -> bytes | None:
        m = self.members.get(path)
        return self.tar.extractfile(m).read() if m is not None else None

    def text(self, path: str) -> str | None:
        b = self.read(path)
        return b.decode("utf-8", "replace") if b is not None else None

    def match(self, pattern: str) -> list[str]:
        if "*" not in pattern:
            return [pattern] if pattern in self.members else []
        return sorted(p for p in self.members if fnmatch.fnmatchcase(p, pattern))

    def info(self, path: str) -> dict:
        """파일 크기와 (가능하면) 모양: .npy 의 shape·dtype, .npz 의 배열마다 shape, .csv 의 행·열 수."""
        m = self.members.get(path)
        if m is None:
            return {}
        out = dict(size=m.size)
        try:
            if path.endswith(".npy"):
                out.update(_npy_header(self.tar.extractfile(m)))
            elif path.endswith(".npz"):
                with zipfile.ZipFile(io.BytesIO(self.read(path))) as z:
                    out["arrays"] = {n.removesuffix(".npy"): _npy_header(z.open(n)).get("shape") for n in z.namelist()[:50]}
            elif path.endswith((".csv", ".tsv")) and m.size < 50 * 2**20:
                data = self.read(path)
                head = data.split(b"\n", 1)[0].decode("utf-8", "replace")
                out.update(rows=max(data.count(b"\n") - 1 + (not data.endswith(b"\n")), 0),
                           cols=len(head.split("\t" if path.endswith(".tsv") else ",")))
        except Exception as e:  # noqa: BLE001 (모양을 못 읽어도 계보는 냄)
            out["shape_error"] = type(e).__name__
        return out


def _npy_header(f) -> dict:
    import numpy as np

    version = np.lib.format.read_magic(f)
    shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                             else np.lib.format.read_array_header_2_0(f))
    return dict(shape=list(shape), dtype=str(dtype))


def exit_ok(code) -> bool | None:
    """종료 코드 → 성공 (True) · 실패 (False) · 모름 (None: 끝나기 전에 신호로 끊김. 124 timeout · 137 SIGKILL · 143 SIGTERM,
    Codex 의 명령 시간 제한). 끊긴 실행도 끊기기 전에 파일을 썼을 수 있어 쓰기 후보로 셈 (2026-11-16 physio × Codex 의
    metric_local.py · full_neural.py: exit 143 인데 결과 npz 가 남고 최종 섞기가 읽음)."""
    if code == 0:
        return True
    return None if code is None or code in (124, 137, 143, -9, -15) else False


def harness_runs(d: Path, harness: str, ws: Workspace) -> tuple[list[Run], list[FileEvent]]:
    """하네스 기록 → (실행, 파일 사건), 기록 순서대로."""
    runs, events = [], []
    read_file = ws.text
    if harness == "codex":
        idx = 0
        for line in (d / "codex.jsonl").read_text(errors="replace").splitlines():
            try:
                o = json.loads(line)
            except ValueError:
                continue
            it = o.get("item") or {}
            if o.get("type") != "item.completed":
                continue
            if it.get("type") == "command_execution":
                idx += 1
                r, e, _ = parse_shell(unwrap_bash(it.get("command", "")), WORK, idx, exit_ok(it.get("exit_code")), read_file)
                runs += r
                events += e
            elif it.get("type") == "file_change":  # apply_patch 등 (내용은 기록에 없음)
                idx += 1
                for ch in it.get("changes") or []:
                    if ch.get("path"):
                        events.append(FileEvent((idx, 0), _abs(ch["path"], WORK), None, "file_change"))
    elif harness == "claude-code":
        rows = []
        for line in (d / "claude-code.jsonl").read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        failed = set()
        for o in rows:
            if o.get("type") == "user":
                for b in (o.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                        failed.add(b.get("tool_use_id"))
        idx, cwd = 0, WORK  # Claude Code 의 Bash 는 cwd 가 다음 명령으로 이어짐
        for o in rows:
            if o.get("type") != "assistant":
                continue
            for b in (o.get("message") or {}).get("content") or []:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                idx += 1
                name, inp, ok = b.get("name"), b.get("input") or {}, b.get("id") not in failed
                if name == "Bash":
                    r, e, cwd2 = parse_shell(inp.get("command", ""), cwd, idx, ok, read_file)
                    runs += r
                    events += e
                    cwd = cwd2 if cwd2.startswith(WORK) else WORK
                elif name == "Write" and inp.get("file_path") and ok:
                    events.append(FileEvent((idx, 0), _abs(inp["file_path"], cwd), inp.get("content", ""), "Write"))
                elif name == "Edit" and inp.get("file_path") and ok:
                    events.append(FileEvent((idx, 0), _abs(inp["file_path"], cwd), None, "Edit",
                                            (inp.get("old_string", ""), inp.get("new_string", ""), inp.get("replace_all"))))
                elif name == "MultiEdit" and inp.get("file_path") and ok:
                    for k, ed in enumerate(inp.get("edits") or []):
                        events.append(FileEvent((idx, k), _abs(inp["file_path"], cwd), None, "Edit",
                                                (ed.get("old_string", ""), ed.get("new_string", ""), ed.get("replace_all"))))
                elif name == "NotebookEdit" and inp.get("notebook_path"):
                    events.append(FileEvent((idx, 0), _abs(inp["notebook_path"], cwd), None, "NotebookEdit"))
    return runs, events


def attach_code(runs: list[Run], events: list[FileEvent], ws: Workspace):
    """script 실행마다 실행 시점의 코드 (위 머리말 2 의 '실행 시점의 코드')."""
    by_path: dict[str, list[FileEvent]] = {}
    for e in sorted(events, key=lambda e: e.order):
        by_path.setdefault(e.path, []).append(e)
    for r in runs:
        if r.kind != "script":
            continue
        evs = by_path.get(r.path, [])
        content = None
        for e in evs:
            if e.order >= r.order:
                break
            if e.content is not None:
                content = e.content
            elif e.edit is not None and content is not None and e.edit[0] in content:
                old, new, all_ = e.edit
                content = content.replace(old, new) if all_ else content.replace(old, new, 1)
            else:
                content = None
        later = [e for e in evs if e.order > r.order]
        if content is not None:
            r.code, r.version = content, "log"
        elif (t := ws.text(r.path)) is not None:
            r.code, r.version, r.changed_after_run = t, "workspace", bool(later)
        else:
            r.version = "missing"


def aide_runs(ws: Workspace) -> tuple[list[Run], str | None, list[dict]]:
    """AIDE journal → 노드마다 실행 하나 (step 순), 최종 노드 id, 노드 요약."""
    j = ws.text(f"{WORK}/logs/exp/journal.json")
    if j is None:
        return [], None, []
    journal = json.loads(j)
    nodes = journal.get("nodes", [])
    node2parent = journal.get("node2parent") or {}  # 노드의 parent 칸은 비어 있고 관계는 여기 (AIDE journal)
    best = ws.text(f"{AIDE_EXP}/best_solution/node_id.txt")
    best = best.strip() if best else None
    if best is None and (code := ws.text(f"{WORK}/logs/exp/best_solution.py")) is not None:
        best = next((n["id"] for n in nodes if n.get("code") == code), None)
    runs, summary = [], []
    for n in sorted(nodes, key=lambda n: int(n.get("step", 0))):
        ok = not n.get("is_buggy") and n.get("exc_type") in (None, "None")
        runs.append(Run((int(n.get("step", 0)), 0), "aide_node", f"AIDE node {n['id']} (step {n.get('step')})", ok,
                        AIDE_EXP, code=n.get("code"), version="journal", node_id=n["id"]))
        metric = n.get("metric")
        parent = n.get("parent") if n.get("parent") not in (None, "None") else node2parent.get(n["id"])
        summary.append(dict(id=n["id"], step=int(n.get("step", 0)), parent=parent, buggy=bool(n.get("is_buggy")),
                            metric=metric.get("value") if isinstance(metric, dict) else metric))
    return runs, best, summary


# ---------- 계보 ----------

def _spec(pattern: str) -> int:
    """경로의 구체성 = 파일 이름에서 * 밖 글자 수."""
    return len(posixpath.basename(pattern).replace("*", ""))


def _stem_spec(pattern: str) -> int:
    """파일 이름 (확장자 뺌) 의 고정 글자 수. 0 이면 이름이 통째로 변수 (쓰기로도 안 셈)."""
    return len(posixpath.splitext(posixpath.basename(pattern))[0].replace("*", ""))


def _role_of(path: str) -> str | None:
    """원본 데이터 (따라가지 않음): HOME/data · AIDE 의 ./input 아래, 또는 이름이 train.csv 등 (폴더를 몰라 */train.csv 여도)."""
    base = posixpath.basename(path)
    if path.startswith(DATA + "/") or path.startswith(AIDE_EXP + "/input/"):
        return "data"
    if base in DATA_NAMES or (base.startswith("*") and base.lstrip("*") in DATA_NAMES):
        return "data"
    return None


class Lineage:
    def __init__(self, d: Path):
        self.d = d
        self.meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        self.harness = self.meta.get("harness") or next(
            (h for h in ("codex", "claude-code", "aide") if (d / f"{h}.jsonl").exists()), None)
        self.ws = Workspace(d / "workspace.tgz")
        self.aide_best, self.aide_nodes = None, []
        if self.harness == "aide":
            self.runs, self.aide_best, self.aide_nodes = aide_runs(self.ws)
            self.events = []
        else:
            self.runs, self.events = harness_runs(d, self.harness, self.ws)
            attach_code(self.runs, self.events, self.ws)
        # 실행마다 캐시. 값에 실행 객체를 같이 둠 (잠깐 만든 exec·모듈 Run 이 사라진 뒤 같은 id 가 다시 쓰이지 않게)
        self._refs: dict[int, tuple[Run, list[Ref] | None]] = {}
        self._writes: dict[int, tuple[Run, list[str]]] = {}
        self.nodes: dict[str, dict] = {}
        self.unresolved: list[dict] = []
        self._keep: list[Run] = []  # 잠깐 만든 Run 을 붙잡아 둠 (id 캐시)
        self._file_refs: dict[str, list[Ref]] = {}  # 작업 디렉터리 .py 의 경로 (exec·모듈, 파일마다 한 번)
        self._exec_cache: dict[int, tuple[Run, list[str]]] = {}

    # 실행의 쓰기 경로 (절대, 패턴)
    def refs(self, r: Run) -> list[Ref]:
        if id(r) not in self._refs:
            self._refs[id(r)] = (r, code_refs(r.code) if r.code else None)
        return self._refs[id(r)][1] or []

    def local_modules(self, r: Run) -> list[tuple[str, str]]:
        """(모듈 이름, 절대 경로): import 하는 모듈 중 작업 디렉터리에 있는 것 (스크립트 폴더 → cwd → WORK 순)."""
        if not r.code:
            return []
        dirs = [posixpath.dirname(r.path)] if r.path else []
        dirs += [r.cwd, WORK]
        out = []
        for m in code_imports(r.code):
            name = m.lstrip(".")
            for base in dict.fromkeys(dirs):
                for cand in (f"{base}/{name}.py", f"{base}/{name}/__init__.py"):
                    if cand in self.ws.members:
                        out.append((m, cand))
                        break
                else:
                    continue
                break
        return out

    def module_refs(self, r: Run, depth: int = 0) -> list[tuple[str, Ref]]:
        """로컬 모듈의 경로 (모듈 이름, Ref). 모듈의 상대 경로는 실행 cwd 기준 (파이썬 규칙)."""
        out = []
        for name, path in self.local_modules(r):
            code = self.ws.text(path)
            out += [(name, ref) for ref in self.file_refs(path)]
            if depth < 3:
                sub = Run(r.order, "module", "", r.ok, r.cwd, path=path, code=code, version="workspace")
                out += self.module_refs(sub, depth + 1)
        return out

    def exec_deps(self, r: Run, depth: int = 0) -> list[str]:
        """이 실행 안에서 돌린 다른 .py (절대 경로): 코드에 exec( · compile( · runpy 가 있고 그 파일을 읽음
        (exec(open('x.py').read()) · s = Path('x.py').read_text().replace(...); exec(compile(s, ...))). 그 파일의 쓰기는
        이 실행의 쓰기 (2026-11-18 physio × Codex: 최종 제출을 쓴 #272 가 compose_submission_nn.py 의 무게 식을 바꿔 exec)."""
        if not r.code or not re.search(r"\bexec\s*\(|\bcompile\s*\(|\brunpy\b", r.code) or depth > 3:
            return []
        if id(r) in self._exec_cache:
            return self._exec_cache[id(r)][1]
        out, lines = [], r.code.split("\n")
        for ref in self.refs(r):
            path = _abs(ref.pattern, r.cwd)
            if ref.role == "read" and path.endswith(".py") and "*" not in path and path in self.ws.members:
                src = lines[ref.line - 1] if 0 < ref.line <= len(lines) else ""
                if PARTIAL.search(src[ref.col:]):
                    # 읽은 글을 잘라 앞부분만 돌림 (exec(open('r.py').read().split('m=CatBoost')[0])) → 그 파일의 쓰기는 안 셈
                    continue
                out.append(path)
                sub = Run(r.order, "exec", "", r.ok, r.cwd, path=path, code=self.ws.text(path), version="workspace")
                self._keep.append(sub)
                out += self.exec_deps(sub, depth + 1)
        self._exec_cache[id(r)] = (r, list(dict.fromkeys(out)))
        return self._exec_cache[id(r)][1]

    def file_refs(self, path: str) -> list[Ref]:
        if path not in self._file_refs:
            self._file_refs[path] = code_refs(self.ws.text(path) or "") or []
        return self._file_refs[path]

    def writes(self, r: Run) -> list[str]:
        """실행이 쓰는 경로 (절대 패턴). 로컬 모듈과 exec 로 돌린 코드의 쓰기도 (실행 cwd 기준)."""
        if id(r) not in self._writes:
            if r.kind == "copy":
                out = [r.path]
            else:
                refs = self.refs(r) + [ref for _, ref in self.module_refs(r)]
                for path in self.exec_deps(r):
                    refs += self.file_refs(path)
                out = list(r.writes) + [_abs(ref.pattern, r.cwd) for ref in refs if ref.role == "write"]
            self._writes[id(r)] = (r, out)
        return self._writes[id(r)][1]

    def node_key(self, r: Run) -> str:
        if r.kind == "aide_node":
            return f"aide:{r.node_id}"
        if r.kind in ("script", "exec"):
            return r.path
        return f"{r.kind}#{r.order[0]}.{r.order[1]}"

    def producers(self, target: str, before: tuple, need_ok: bool = True) -> tuple[str, list[Run], list[Run]]:
        """target (절대 경로) 을 쓴, before 보다 앞선 실행 → (status, 실행, 뒤에 온 와일드카드 쓰기).
        정확한 경로 (* 없음) 로 쓴 실행이 있으면 그중 마지막 (덮어쓰기). 없으면 * 밖 글자가 가장 많은 쓰기의 마지막,
        그 쓰기가 서로 다른 코드 둘 이상이면 ambiguous (후보만, 고르지 않음). 못 찾으면 not_found."""
        exact, wild = [], []
        for r in self.runs:
            if r.order >= before or (need_ok and r.ok is False):
                continue
            ws = self.writes(r)
            if target in ws:
                exact.append(r)
            elif (spec := max((_spec(w) for w in ws if "*" in w and _stem_spec(w) and fnmatch.fnmatchcase(target, w)),
                              default=-1)) >= 0:
                wild.append((spec, r))
        if exact:
            last = max(exact, key=lambda r: r.order)
            later = [r for _, r in wild if r.order > last.order]
            return "found", [last], later
        if not wild:
            return "not_found", [], []
        top = max(s for s, _ in wild)
        best = [r for s, r in wild if s == top]
        keys = list(dict.fromkeys(self.node_key(r) for r in best))
        if len(keys) > 1:
            last = {k: max((r for r in best if self.node_key(r) == k), key=lambda r: r.order) for k in keys}
            return "ambiguous", sorted(last.values(), key=lambda r: r.order), []
        return "found", [max(best, key=lambda r: r.order)], []

    def final_run(self) -> tuple[str, Run | None, str]:
        """(제출 경로, 그 파일을 마지막으로 쓴 실행, 상태)."""
        if self.harness == "aide":
            r = next((r for r in self.runs if r.node_id == self.aide_best), None)
            return self.meta.get("submission_from") or f"{AIDE_EXP}/best_submission/submission.csv", r, \
                "found" if r else "not_found"
        sub = self.meta.get("submission_from") or f"{HOME}/submission/submission.csv"
        status, rs, later = self.producers(sub, (10**9,))
        self.final_later = later
        return sub, (rs[-1] if status == "found" else None), status

    def trace(self, r: Run, depth: int = 0, max_depth: int = 25) -> str:
        key = self.node_key(r)
        if key in self.nodes:  # 같은 코드는 한 번 (실행이 여럿이면 executions 에 모음)
            n = self.nodes[key]
            if r.order not in [tuple(e["order"]) for e in n["executions"]]:
                n["executions"].append(self._exec(r))
            return key
        node = dict(id=key, kind=r.kind, file=self._ws_name(r.path) if r.kind in ("script", "exec") else None,
                    version=r.version, changed_after_run=r.changed_after_run, executions=[self._exec(r)],
                    imports=code_imports(r.code) if r.code else [],
                    local_modules=[dict(module=m, file=self._ws_name(p)) for m, p in self.local_modules(r)],
                    reads=[], writes=sorted(set(self._ws_name(w) for w in self.writes(r))))
        if r.kind == "aide_node":
            node["aide_ancestry"] = self._ancestry(r.node_id)
        if (deps := self.exec_deps(r)):
            node["exec"] = [self._ws_name(x) for x in deps]
            if r.code and ".replace(" in r.code:
                # 끌어온 코드를 문자열로 고쳐 돌렸을 수 있음 → 끌어온 파일 그대로가 실행한 코드가 아님 (카드에서 확인)
                node["exec_text_modified"] = True
        if r.code and id(r) in self._refs and self._refs[id(r)][1] is None:
            node["parse_error"] = True
        self.nodes[key] = node
        if r.kind == "copy":
            node["reads"].append(self._read(r, r.src, "copy", None, depth, max_depth))
            return key
        refs = [(None, ref) for ref in self.refs(r)] + [(m, ref) for m, ref in self.module_refs(r)]
        own_writes, seen = self.writes(r), set()
        # 이 코드 안에서 경로마다 첫 쓰기 줄 (모듈의 쓰기는 줄을 견줄 수 없어 뺌). 읽기가 그보다 앞이면 읽고 고쳐 다시 쓰기
        first_write = {}
        for mod, ref in refs:
            if ref.role == "write" and mod is None:
                w = _abs(ref.pattern, r.cwd)
                first_write[w] = min(first_write.get(w, (ref.line, ref.col)), (ref.line, ref.col))
        self._first_write = first_write
        for mod, ref in refs:
            path = _abs(ref.pattern, r.cwd)
            if ref.role == "write" or (path, mod) in seen or (_role_of(path) == "data" and path in seen):
                continue
            if not (_is_file(path) or path.endswith(".py")):  # 확장자 없는 이름 (반복 목록의 키 등) 은 파일 경로로 안 봄
                continue
            seen |= {(path, mod), path}
            node["reads"].append(self._read(r, _abs(ref.pattern, r.cwd), ref.role, mod, depth, max_depth,
                                            own_writes, ref.line, ref.col))
        return key

    def _read(self, r: Run, pattern: str, role: str, module, depth, max_depth, own_writes=(), line=0, col=0) -> dict:
        item = dict(pattern=self._ws_name(pattern), role=role, line=line)
        if module:
            item["module"] = module
        if _role_of(pattern) == "data":
            item["status"] = "data"
            return item
        if role == "mention" and pattern.endswith("submission.csv"):
            item["status"] = "submission"  # 제출 경로를 글자로만 씀 (print·주석 글 등, 읽기 호출이 아님)
            return item
        if pattern.endswith(".py") and "*" not in pattern and r.kind != "copy":
            # exec(open('x.py').read()) · runpy: 다른 스크립트의 코드를 이 실행 안에서 돌림 → 코드 의존으로 따라감
            item["status"] = "code"
            code = self.ws.text(pattern)
            if code is None:
                item["status"] = "not_found"
                item["reason"] = "끌어온 코드가 작업 디렉터리에 없음"
                self.unresolved.append(dict(node=self.node_key(r), pattern=item["pattern"], reason=item["reason"]))
                return item
            sub = Run(r.order, "exec", r.command, r.ok, r.cwd, path=pattern, code=code, version="workspace")
            item["node"] = self.trace(sub, depth + 1, max_depth) if depth < max_depth else self.node_key(sub)
            return item
        rewrites = False
        if role != "copy" and any(pattern == w or fnmatch.fnmatchcase(pattern, w) for w in own_writes):
            # 같은 코드가 쓰고 읽음. 이 실행보다 앞서 다른 코드가 쓴 판이 있으면 읽고 고쳐 다시 쓰기 (read-modify-write,
            # 예: 제출을 읽어 보정하고 덮어씀) → 그 생산자로 이어 감. 없으면 임시 파일·캐시 (self)
            # (2026-11-18 QSAR_fish × Codex: 제출을 세 번 덮어쓴 명령이 self 로 멈춰 앞 계보가 빠졌음)
            status, rs, _ = self.producers(pattern, r.order) if "*" not in pattern else ("not_found", [], [])
            # 같은 코드 안에서 쓰기보다 앞 줄에서 읽을 때만 (쓴 뒤 검사로 다시 읽는 것은 self: AIDE 노드가 제출을 쓰고
            # 검사 서버에 보내려고 open(..., 'rb') 로 읽음). 셸 리다이렉트·복사로만 쓰면 줄이 없어 self
            writes_at = [ln for w, ln in getattr(self, "_first_write", {}).items()
                         if w == pattern or fnmatch.fnmatchcase(pattern, w)]
            before = module is None and line and writes_at and (line, col) < min(writes_at)
            if not before or not [p for p in rs if self.node_key(p) != self.node_key(r)]:
                item["status"] = "self"
                return item
            rewrites = item["rewrites"] = True
        if "*" in pattern and not posixpath.splitext(posixpath.basename(pattern))[0].replace("*", ""):
            item["status"] = "generic"  # 파일 이름이 통째로 변수 (f'{name}.npz') → 맞는 파일·생산자를 정할 수 없음
            item["reason"] = "파일 이름에 고정 글자가 없음 (변수로만 정해짐)"
            item["matches"] = len(self.ws.match(pattern))
            self.unresolved.append(dict(node=self.node_key(r), pattern=item["pattern"], reason=item["reason"]))
            return item
        files = self.ws.match(pattern)
        self_written = [f for f in files if any(fnmatch.fnmatchcase(f, w) for w in own_writes)]
        if files and len(self_written) == len(files) and role != "copy" and not rewrites:
            item["status"] = "self"  # 같은 코드가 쓰고 읽음 (캐시 등)
            item["files"] = [self._ws_name(f) for f in files]
            return item
        targets = files or ([pattern] if "*" not in pattern else [])
        item["files"] = [dict(file=self._ws_name(f), **self.ws.info(f)) for f in files[:200]]
        if len(files) > 200:
            item["files_more"] = len(files) - 200
        if not targets:
            item["status"] = "not_found"
            item["reason"] = "작업 디렉터리에 맞는 파일 없음 (경로에 모르는 부분)"
            self.unresolved.append(dict(node=self.node_key(r), pattern=item["pattern"], reason=item["reason"]))
            return item
        if not files:
            packed = pattern.startswith(WORK + "/") or pattern.startswith(f"{HOME}/submission/")
            item["note"] = ("회수물에 없음 (20MB 초과로 빠졌거나 지움)" if packed else
                            "작업 디렉터리 밖 (회수 범위 아님: WORK 와 HOME/submission 만 묶음)")
        producers, statuses = {}, set()
        later_wild = {}
        for t in targets:
            status, rs, later = self.producers(t, r.order)
            statuses.add(status)
            for p in rs:
                if self.node_key(p) != self.node_key(r):  # 자기 앞 실행 (같은 코드) 은 계보가 아님
                    producers.setdefault(self.node_key(p), (status, p))
            for p in later:
                later_wild.setdefault(self.node_key(p), p)
        item["status"] = "found" if statuses == {"found"} else "ambiguous" if "ambiguous" in statuses else \
            "partial" if "found" in statuses else "not_found"
        item["producers"] = []
        for k, (status, p) in producers.items():
            entry = dict(node=k, status=status, order=list(p.order))
            if status == "found" and depth < max_depth:
                self.trace(p, depth + 1, max_depth)
            item["producers"].append(entry)
        if later_wild:  # 정확한 경로의 마지막 쓰기 뒤에 같은 파일과 맞는 * 경로로 쓴 실행 (덮어썼을 수 있음, 카드에서 확인)
            item["later_wildcard_writers"] = list(later_wild)
        if item["status"] != "found":
            reason = dict(not_found="쓰는 실행을 못 찾음", ambiguous="가장 구체적인 쓰기가 여러 코드",
                          partial="일부 파일만 쓰는 실행을 찾음")[item["status"]]
            item["reason"] = reason
            self.unresolved.append(dict(node=self.node_key(r), pattern=item["pattern"], reason=reason,
                                        candidates=[k for k in producers]))
        return item

    def _ancestry(self, node_id):
        by_id = {n["id"]: n for n in self.aide_nodes}
        out, cur = [], by_id.get(node_id)
        while cur is not None and len(out) < 200:
            out.append(dict(id=cur["id"], step=cur["step"], metric=cur["metric"]))
            cur = by_id.get(cur["parent"]) if cur["parent"] not in (None, "None") else None
        return out

    def _exec(self, r: Run) -> dict:
        out = dict(order=list(r.order), ok=r.ok, cwd=self._ws_name(r.cwd), command=r.command[:2000])
        if r.env:
            out["env"] = r.env  # 스레드 수 같은 것은 뺌. 코드가 os.environ 으로 판을 고르면 여기서 봄
        return out

    @staticmethod
    def _ws_name(p: str | None) -> str | None:
        if p is None:
            return None
        if p == WORK or p.startswith(WORK + "/"):
            return "workspace" + p[len(WORK):]
        return p

    def result(self) -> dict:
        sub, r, status = self.final_run()
        recovered = self.d / "submission.csv"
        copy = self.ws.read(sub)
        out = dict(key=self.meta.get("key") or self.d.name, harness=self.harness,
                   submission=dict(path=sub, status=status, same_content=self.same_content(),
                                   # 회수한 submission.csv 와 작업 디렉터리에 묶인 제출 사본 (sub) 이 바이트까지 같은지 (없으면 None)
                                   matches_recovered=(recovered.read_bytes() == copy) if recovered.exists() and copy is not None
                                   else None))
        if r is not None:
            out["final"] = self.trace(r)
            if getattr(self, "final_later", None):
                out["submission"]["later_wildcard_writers"] = [self.node_key(x) for x in self.final_later]
        else:
            cands = [self.node_key(x) for x in self.runs if any("submission" in w for w in self.writes(x))]
            out["final"] = None
            self.unresolved.append(dict(node=None, pattern=sub, reason=f"제출을 쓴 실행: {status}",
                                        candidates=list(dict.fromkeys(cands))[-10:]))
        if self.harness == "aide":
            out["aide"] = dict(best=self.aide_best, nodes=len(self.aide_nodes))
        out["nodes"] = list(self.nodes.values())
        out["unresolved"] = self.unresolved
        out["counts"] = dict(runs=len(self.runs), events=len(self.events), workspace_files=len(self.ws.members))
        return out

    def same_content(self) -> list[str]:
        """작업 디렉터리에서 submission.csv 와 바이트가 같은 파일."""
        p = self.d / "submission.csv"
        if not p.exists() or self.ws.tar is None:
            return []
        data = p.read_bytes()
        sub = self.meta.get("submission_from")
        return sorted(self._ws_name(k) for k, m in self.ws.members.items()
                      if k != sub and m.size == len(data) and self.ws.read(k) == data)


# ---------- 출력 ----------

def render(res: dict) -> str:
    """사람이 읽는 나무 (최종 → 거꾸로)."""
    nodes = {n["id"]: n for n in res["nodes"]}
    same = {True: ", 회수한 제출과 같음", False: ", 회수한 제출과 다름", None: ""}[res["submission"].get("matches_recovered")]
    lines = [f"{res['key']} ({res['harness']})",
             f"제출 {res['submission']['path']} ← {res['final'] or '못 찾음'} ({res['submission']['status']}{same})"]
    if res["submission"]["same_content"]:
        lines.append(f"  같은 내용: {', '.join(res['submission']['same_content'])}")
    seen = set()

    def walk(nid, indent):
        n = nodes.get(nid)
        pad = "  " * indent
        if n is None:
            lines.append(f"{pad}- {nid}")
            return
        name = n["file"] or nid
        flag = f" 판={n['version']}" + (" (실행 뒤 바뀜)" if n["changed_after_run"] else "")
        ex = n["executions"]
        env = " " + " ".join(f"{k}={v}" for k, v in ex[0]["env"].items()) if ex[0].get("env") else ""
        lines.append(f"{pad}- {name} [{n['kind']}{flag}] 실행 {len(ex)}번, 따라간 것 #{ex[0]['order'][0]}{env}")
        if nid in seen:
            return
        seen.add(nid)
        if n["imports"]:
            lines.append(f"{pad}    import: {' '.join(n['imports'])}")
        if n["local_modules"]:
            lines.append(f"{pad}    로컬 모듈: {', '.join(m['file'] for m in n['local_modules'])}")
        if n.get("exec"):
            mod = " (글자를 고쳐 돌렸을 수 있음)" if n.get("exec_text_modified") else ""
            lines.append(f"{pad}    exec: {', '.join(n['exec'])}{mod}")
        for rd in n["reads"]:
            files = rd.get("files") or []
            cnt = f" ({len(files) + rd.get('files_more', 0)}개)" if len(files) > 1 else ""
            tag = {"data": "데이터", "submission": "제출 언급", "code": "코드", "self": "자기가 씀", "generic": "이름이 변수", "not_found": "못 찾음",
                   "ambiguous": "여럿", "partial": "일부만", "found": ""}.get(rd["status"], rd["status"])
            prods = [p["node"] for p in rd.get("producers", [])]
            arrow = f" ← {', '.join(nodes[p]['file'] or p if p in nodes else p for p in prods)}" if prods else ""
            rw = " (읽고 고쳐 다시 씀)" if rd.get("rewrites") else ""
            lines.append(f"{pad}    읽음 {rd['pattern']}{cnt} [{rd['role']}] {tag}{rw}{arrow}".rstrip())
            if rd["status"] == "code" and rd.get("node"):
                walk(rd["node"], indent + 3)
            for p in rd.get("producers", []):
                if p["status"] == "found":
                    walk(p["node"], indent + 3)

    if res["final"]:
        walk(res["final"], 1)
    if res["unresolved"]:
        lines.append("못 찾은 연결:")
        for u in res["unresolved"]:
            c = f" 후보 {', '.join(u['candidates'])}" if u.get("candidates") else ""
            lines.append(f"  - {u['node']}: {u['pattern']} ({u['reason']}){c}")
    return "\n".join(lines)


def run_dir(target: str) -> Path:
    p = Path(target).expanduser()
    if p.is_dir():
        return p
    from fab.agent import AGENT_RUNS

    if (AGENT_RUNS / target).is_dir():
        return AGENT_RUNS / target
    raise SystemExit(f"실행 폴더 없음: {target} (AGENT_RUNS={AGENT_RUNS}, 받기: fab sync --pull outputs/_mlebench/agents/{target})")


def main(argv: list[str]):
    ap = argparse.ArgumentParser(prog="fab ta agent lineage", description="최종 제출의 계보 (fab/agent/lineage.py)")
    ap.add_argument("target", help="실행 key (AGENT_RUNS 아래) 또는 실행 폴더")
    ap.add_argument("--json", action="store_true", help="JSON 으로 (기본은 나무)")
    ap.add_argument("--out", help="JSON 을 이 파일에 씀")
    args = ap.parse_args(argv)
    res = Lineage(run_dir(args.target)).result()
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    if args.json and not args.out:
        json.dump(res, sys.stdout, ensure_ascii=False, indent=1)
        print()
    else:
        print(render(res))
