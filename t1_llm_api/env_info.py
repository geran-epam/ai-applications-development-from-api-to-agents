import json, os, subprocess, sys, glob

paths = [
    os.path.expanduser("~/.claude/settings.json"),
    os.path.expanduser("~/.claude/settings.local.json"),
    os.path.expanduser("~/.claude.json"),
    os.path.expanduser("~/.claude-code-router/config.json"),
]

print("=== JSON Validation ===")
for p in paths:
    if os.path.exists(p):
        try:
            json.load(open(p))
            print(f"[OK]    {p}")
        except json.JSONDecodeError as e:
            print(f"[FAIL]  {p} -> {e}")
    else:
        print(f"[MISS]  {p}")

print("\n=== CCR process ===")
subprocess.run("ps aux | grep -i claude-code-router | grep -v grep", shell=True)

print("\n=== Port 3456 listener ===")
subprocess.run("ss -ltnp 2>/dev/null | grep 3456 || netstat -ltnp 2>/dev/null | grep 3456", shell=True)

print("\n=== CCR health check ===")
subprocess.run('curl -s -o /dev/null -w "HTTP %{http_code}\\n" http://127.0.0.1:3456/v1/models', shell=True)
subprocess.run('curl -s http://127.0.0.1:3456/ 2>&1 | head -c 500', shell=True)

print("\n\n=== Claude Code debug env check ===")
for var in ["ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_MODEL"]:
    print(var, "=", os.environ.get(var, "<not set in current shell>"))