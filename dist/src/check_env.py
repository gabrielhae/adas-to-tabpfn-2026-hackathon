import torch, sys, platform
print("python     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("cuda avail :", torch.cuda.is_available())
print("cuda ver   :", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu        :", torch.cuda.get_device_name(0))
    print("gpu mem GB :", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    print("capability :", torch.cuda.get_device_capability(0))

print("\n--- backend quick test ---")
x = torch.randn(512, 512, device="cuda" if torch.cuda.is_available() else "cpu")
print("matmul ok  :", (x @ x).shape)

print("\n" + "=" * 70)
print("TABPFN API INSPECTION")
print("=" * 70)
import tabpfn
print("tabpfn version:", getattr(tabpfn, "__version__", "?"))
print("exports:", [n for n in dir(tabpfn) if not n.startswith("_")])

from tabpfn import TabPFNClassifier, TabPFNRegressor
print("\nTabPFNClassifier signature:")
import inspect
print(" ", inspect.signature(TabPFNClassifier.__init__))
print("\nTabPFNClassifier.fit:")
print(" ", inspect.signature(TabPFNClassifier.fit))
print("\nTabPFNClassifier.predict_proba:")
print(" ", inspect.signature(TabPFNClassifier.predict_proba))
try:
    print("\nTabPFNClassifier.predict_proba kwargs / doc:")
    d = (TabPFNClassifier.predict_proba.__doc__ or "")[:900]
    print(d)
except Exception as e:
    print("  err", e)

print("\n--- constant / device helpers ---")
for n in ["ModelVersion", "TabPFNModel", "get_default_device"]:
    print(f"  {n}:", hasattr(tabpfn, n))

print("\n" + "=" * 70)
print("SMOKE TEST: tiny classification fit")
print("=" * 70)
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 8)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
import time
t0 = time.time()
clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
clf.fit(X[:150], y[:150])
p = clf.predict_proba(X[150:])
print(f"  fit+predict ok in {time.time()-t0:.1f}s   proba shape={p.shape}  "
      f"classes={clf.classes_}")
