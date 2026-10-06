import os, re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

base = os.path.join(r"E:\DeepLearningFolder\TabPFN-Hackathon2026", ".venv", "Lib", "site-packages", "tabpfn")
p = os.path.join(base, "browser_auth.py")
t = open(p, encoding="utf-8", errors="ignore").read()
print("browser_auth.py bytes:", len(t))

print("\n=== ENV VARS referenced ===")
envs = set(re.findall(r'(?:getenv|environ\.get|environ\[)\s*[\(\[]\s*["\']([A-Za-z0-9_]+)["\']', t))
for e in sorted(envs):
    print("   ", e)

print("\n=== module-level constants (UPPER) ===")
for m in re.finditer(r"^([A-Z][A-Z0-9_]{2,})\s*=", t, re.M):
    print("   ", m.group(1))

print("\n=== functions ===")
for m in re.finditer(r"^def (\w+)\(", t, re.M):
    print("   ", m.group(1))

print("\n=== references to token file / cache paths ===")
for m in re.finditer(r'["\']([^"\']*(?:token|license|auth)[^"\']*\.(?:json|txt|key))["\']', t):
    print("   ", m.group(1))

print("\n=== HF repo map ===")
for f in ["model_loading.py"]:
    q = os.path.join(base, f)
    if os.path.exists(q):
        s = open(q, encoding="utf-8", errors="ignore").read()
        for m in re.finditer(r'_HF_REPOS\s*[:=]\s*\{(.*?)\}', s, re.S):
            print("   ", " ".join(m.group(1).split())[:600])

print("\n=== search whole package for env-var escape hatches ===")
for root, dirs, files in os.walk(base):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        fp = os.path.join(root, fn)
        try:
            s = open(fp, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for m in re.finditer(r'(?:getenv|environ\.get)\s*[\(\[]\s*["\']([A-Za-z0-9_]+)["\']', s):
            print(f"   {fn}: {m.group(1)}")
