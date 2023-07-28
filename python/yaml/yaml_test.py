import yaml
from pathlib import Path

yaml_dict = yaml.safe_load(Path("data.yml").read_text())
print(yaml_dict['lr'])