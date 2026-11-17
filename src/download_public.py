"""Retrieve the exact official SpotLight sample inputs used in this submission."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

HASHES={'platform.csv':'4b17ed8082726385efd4c6cbf25e845264ecc8efb9817c2ae644ca511a81cae5','label':'2cb42f37b2cec0dffd04f1a3135b9a33c77e1c9305d6c567ccaf218281335f83'}
if __name__=='__main__':
 p=Path(__file__).resolve().parents[1]/'data/spotlight';p.mkdir(parents=True,exist_ok=True)
 for name,h in HASHES.items():
  f=p/name
  if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==h:continue
  data=urlopen('https://raw.githubusercontent.com/netsys-edinburgh/SpotLight/main/sample/'+name,timeout=60).read()
  if hashlib.sha256(data).hexdigest()!=h:raise ValueError('Upstream data changed; retain archived inputs and review provenance before rerunning.')
  f.write_bytes(data)
 print('Official sample hashes verified')
