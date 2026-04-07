import os
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path, 'rb') as fh:
                content = fh.read()
                if b'\x00' in content:
                    print('CORRUPTED:', path)
print("Scan done")