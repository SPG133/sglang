# -*- coding: utf-8 -*-
"""本地 Git Diff 查看器

用法:
    python git_diff_viewer.py [仓库路径]        # 默认当前目录
    然后浏览器打开 http://127.0.0.1:8765

功能:
    - 页面上可切换本地仓库文件夹（需含 .git）
    - 选任意两个提交（或工作区 WORKTREE）对比
    - 文件列表带 M/A/D/R 标记和 +/- 行数统计
    - 点击文件看带着色的逐行 diff
"""
import json
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

MAX_DIFF_BYTES = 800_000  # 单文件 diff 超过此大小时截断

DEFAULT_REPO = sys.argv[1] if len(sys.argv) > 1 else "."


def git(repo: str, *args: str) -> str:
    """在 repo 上执行 git 命令，返回 stdout 文本。"""
    p = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or f"git {' '.join(args)} 失败")
    return p.stdout


# ---------------- 后端数据函数 ----------------

def is_repo(path: str) -> bool:
    try:
        git(path, "rev-parse", "--git-dir")
        return True
    except Exception:
        return False


def get_commits(repo: str, n: int = 300):
    out = git(repo, "log", f"-{n}", "--date=format:%Y-%m-%d %H:%M",
              "--pretty=format:%H%x00%h%x00%ad%x00%an%x00%s")
    commits = []
    for line in out.splitlines():
        parts = line.split("\x00")
        if len(parts) == 5:
            commits.append(dict(hash=parts[0], short=parts[1], date=parts[2],
                                author=parts[3], subject=parts[4]))
    return commits


def _resolve_new_path(composite: str) -> str:
    """把 numstat 的重命名复合路径解析成新路径。
    'a.py => b.py' -> 'b.py';  'dir/{x => y}/f' -> 'dir/y/f'
    """
    if "{" in composite:
        left = composite[:composite.index("{")]
        right = composite[composite.rindex("}") + 1:]
        inner = composite[composite.index("{") + 1:composite.rindex("}")]
        new_seg = inner.split("=>")[1].strip() if "=>" in inner else ""
        return left + new_seg + right
    if " => " in composite:
        return composite.split(" => ")[1]
    return composite


def get_diff_summary(repo: str, a: str, b: str):
    """返回两提交间改动文件列表（含重命名检测与行数统计）。"""
    out = git(repo, "diff", "--name-status", "-M", a, b)
    num = git(repo, "diff", "--numstat", "-M", a, b)
    stats = {}
    for line in num.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            adds, dels, path = parts
            a_i = int(adds) if adds.isdigit() else -1
            d_i = int(dels) if dels.isdigit() else -1
            stats[path] = (a_i, d_i)
            resolved = _resolve_new_path(path)
            if resolved != path:
                stats[resolved] = (a_i, d_i)
    files = []
    for line in out.splitlines():
        parts = line.split("\t")
        if not parts[0]:
            continue
        status = parts[0]
        if status.startswith("R"):
            old, new = parts[1], parts[2]
        else:
            old, new = None, parts[1]
        s = stats.get(new, (0, 0))
        files.append(dict(status=status[0], old=old, new=new,
                          adds=s[0], dels=s[1]))
    return files


def get_file_diff(repo: str, a: str, b: str, path: str) -> str:
    out = git(repo, "diff", "-M", a, b, "--", path)
    if len(out.encode("utf-8", "replace")) > MAX_DIFF_BYTES:
        out = out[:MAX_DIFF_BYTES] + "\n\n... diff 过大已截断 ..."
    return out


# ---------------- 前端页面 ----------------

PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Git Diff 查看器</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font: 13px/1.5 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         background: #1e1e2e; color: #cdd6f4; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 14px; background: #181825; border-bottom: 1px solid #313244;
           display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  header input[type=text] { width: 340px; background: #313244; color: #cdd6f4;
           border: 1px solid #45475a; border-radius: 6px; padding: 6px 10px; font: 12px Consolas, monospace; }
  select { background: #313244; color: #cdd6f4; border: 1px solid #45475a;
           border-radius: 6px; padding: 6px 8px; max-width: 340px; font-size: 12px; }
  button { background: #89b4fa; color: #1e1e2e; border: none; border-radius: 6px;
           padding: 6px 14px; font-weight: 600; cursor: pointer; }
  button:hover { background: #74c7ec; }
  #main { flex: 1; display: flex; overflow: hidden; }
  #left { width: 320px; border-right: 1px solid #313244; display: flex; flex-direction: column; }
  #fileFilter { margin: 8px; background: #313244; color: #cdd6f4; border: 1px solid #45475a;
           border-radius: 6px; padding: 5px 8px; font-size: 12px; }
  #fileList { flex: 1; overflow-y: auto; }
  .fileRow { padding: 6px 10px; cursor: pointer; border-bottom: 1px solid #181825;
             display: flex; gap: 8px; align-items: center; font-family: Consolas, monospace; font-size: 12px; }
  .fileRow:hover { background: #313244; }
  .fileRow.sel { background: #45475a; }
  .badge { flex: none; width: 16px; height: 16px; border-radius: 4px; font-size: 10px;
           display: inline-flex; align-items: center; justify-content: center; font-weight: 700; }
  .bM { background: #f9e2af; color: #1e1e2e; } .bA { background: #a6e3a1; color: #1e1e2e; }
  .bD { background: #f38ba8; color: #1e1e2e; } .bR { background: #89b4fa; color: #1e1e2e; }
  .path { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .adds { color: #a6e3a1; } .dels { color: #f38ba8; }
  #right { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #diffHeader { padding: 8px 14px; background: #181825; border-bottom: 1px solid #313244;
                font-family: Consolas, monospace; font-size: 12px; color: #a6adc8; }
  #diffBox { flex: 1; overflow: auto; background: #11111b; }
  pre { font: 12px/1.55 Consolas, "Courier New", monospace; padding: 8px 0; }
  .dl { padding: 0 12px 0 0; white-space: pre; }
  .ln { display: inline-block; width: 3.5em; text-align: right; padding-right: 8px;
        color: #585b70; user-select: none; }
  .c { color: #6c7086; background: #181825; }      /* 上下文 */
  .a { color: #a6e3a1; background: #12261e; }      /* + */
  .d { color: #f38ba8; background: #2d1b22; }      /* - */
  .h { color: #89dceb; background: #1b1f2d; }      /* @@ */
  .meta { color: #f5c2e7; }                        /* diff --git 等 */
  #hint { padding: 40px; color: #6c7086; text-align: center; }
  .err { color: #f38ba8; padding: 10px 14px; }
</style>
</head>
<body>
<header>
  <span style="font-weight:700">仓库:</span>
  <input type="text" id="repoPath">
  <button onclick="loadRepo()">加载</button>
  <span style="width:12px"></span>
  <span>From:</span><select id="selA" onchange="loadDiff()"></select>
  <button onclick="swap()" title="交换">⇄</button>
  <span>To:</span><select id="selB" onchange="loadDiff()"></select>
  <span id="stat" style="margin-left:auto;color:#a6adc8"></span>
</header>
<div id="main">
  <div id="left">
    <input type="text" id="fileFilter" placeholder="过滤文件名..." oninput="renderFiles()">
    <div id="fileList"><div id="hint">加载仓库后显示文件列表</div></div>
  </div>
  <div id="right">
    <div id="diffHeader">选择文件查看 diff</div>
    <div id="diffBox"><div id="hint">左侧点击文件</div></div>
  </div>
</div>
<script>
let COMMITS = [], FILES = [], SEL = null;

async function api(ep, params) {
  const q = new URLSearchParams(params).toString();
  const r = await fetch(`/api/${ep}?${q}`);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}
const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

async function init() {
  const d = await api("default");
  document.getElementById("repoPath").value = d.path;
  await loadRepo(true);
}

async function loadRepo(autoselect) {
  const path = document.getElementById("repoPath").value.trim();
  try {
    const j = await api("commits", { path });
    COMMITS = j.commits;
    const opts = [`<option value="WORKTREE">WORKSPACE(工作区未提交)</option>`]
      .concat(COMMITS.map(c =>
        `<option value="${c.hash}">${esc(c.date)} ${c.short} ${esc(c.subject).slice(0,40)}</option>`));
    document.getElementById("selA").innerHTML = opts.join("");
    document.getElementById("selB").innerHTML = opts.join("");
    if (autoselect && COMMITS.length) {
      document.getElementById("selA").value = COMMITS[COMMITS.length>1?1:0].hash;
      document.getElementById("selB").value = COMMITS[0].hash;
    }
    loadDiff();
  } catch (e) { alert("加载失败: " + e.message); }
}

function swap() {
  const a = document.getElementById("selA"), b = document.getElementById("selB");
  [a.value, b.value] = [b.value, a.value];
  loadDiff();
}

async function loadDiff() {
  const path = document.getElementById("repoPath").value.trim();
  const a = document.getElementById("selA").value, b = document.getElementById("selB").value;
  SEL = null;
  document.getElementById("diffHeader").textContent = "选择文件查看 diff";
  document.getElementById("diffBox").innerHTML = `<div id="hint">左侧点击文件</div>`;
  if (a === b) {
    FILES = []; renderFiles();
    document.getElementById("stat").textContent = "From 与 To 相同";
    return;
  }
  try {
    const j = await api("diffsummary", { path, a, b });
    FILES = j.files;
    const adds = FILES.reduce((s,f)=>s+(f.adds>0?f.adds:0),0);
    const dels = FILES.reduce((s,f)=>s+(f.dels>0?f.dels:0),0);
    document.getElementById("stat").textContent =
      `${FILES.length} 个文件  +${adds} −${dels}`;
    renderFiles();
  } catch (e) {
    FILES = []; renderFiles();
    document.getElementById("stat").textContent = "错误: " + e.message;
  }
}

function renderFiles() {
  const q = document.getElementById("fileFilter").value.toLowerCase();
  const box = document.getElementById("fileList");
  const files = FILES.filter(f => (f.new||"").toLowerCase().includes(q));
  if (!files.length) { box.innerHTML = `<div id="hint">无匹配文件</div>`; return; }
  box.innerHTML = files.map(f => {
    const name = f.status === "R" ? `${f.old} → ${f.new}` : f.new;
    return `<div class="fileRow ${SEL===f.new?"sel":""}" onclick="showFile('${esc(f.new).replace(/'/g,"&#39;")}')">
      <span class="badge b${f.status}">${f.status}</span>
      <span class="path" title="${esc(name)}">${esc(name)}</span>
      <span class="adds">+${f.adds>0?f.adds:""}</span><span class="dels">−${f.dels>0?f.dels:""}</span>
    </div>`;
  }).join("");
}

async function showFile(path) {
  SEL = path; renderFiles();
  document.getElementById("diffHeader").textContent = path;
  const box = document.getElementById("diffBox");
  try {
    const j = await api("filediff", {
      path: document.getElementById("repoPath").value.trim(),
      a: document.getElementById("selA").value,
      b: document.getElementById("selB").value,
      file: path,
    });
    if (!j.diff.trim()) {
      box.innerHTML = `<div id="hint">（无文本差异，可能是二进制文件）</div>`;
      return;
    }
    const lines = j.diff.split("\n");
    let n1 = 0, n2 = 0, html = "";
    for (const line of lines) {
      let cls = "c", l1 = "", l2 = "";
      if (line.startsWith("+++") || line.startsWith("---") || line.startsWith("diff ") ||
          line.startsWith("index ") || line.startsWith("new ") || line.startsWith("old ") ||
          line.startsWith("similarity ") || line.startsWith("rename ") || line.startsWith("\\")) {
        cls = "meta";
      } else if (line.startsWith("@@")) {
        const m = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)/);
        if (m) { n1 = +m[1]; n2 = +m[2]; }
        cls = "h";
      } else if (line.startsWith("+")) { cls = "a"; l2 = n2++; }
      else if (line.startsWith("-")) { cls = "d"; l1 = n1++; }
      else { l1 = n1++; l2 = n2++; }
      html += `<div class="dl ${cls}"><span class="ln">${l1}</span><span class="ln">${l2}</span>${esc(line)}</div>`;
    }
    box.innerHTML = `<pre>${html}</pre>`;
  } catch (e) {
    box.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}
init();
</script>
</body>
</html>"""


# ---------------- HTTP 服务 ----------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静默访问日志
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                data = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif u.path == "/api/default":
                self._json({"path": DEFAULT_REPO})
            elif u.path == "/api/commits":
                repo = q.get("path", DEFAULT_REPO)
                if not is_repo(repo):
                    self._json({"error": f"不是 git 仓库: {repo}"})
                else:
                    self._json({"commits": get_commits(repo)})
            elif u.path == "/api/diffsummary":
                self._json({"files": get_diff_summary(q["path"], q["a"], q["b"])})
            elif u.path == "/api/filediff":
                self._json({"diff": get_file_diff(q["path"], q["a"], q["b"], q["file"])})
            else:
                self._json({"error": "unknown"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)


def main():
    port = 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Git Diff 查看器已启动: {url}  (仓库默认: {DEFAULT_REPO})")
    print("按 Ctrl+C 停止")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
