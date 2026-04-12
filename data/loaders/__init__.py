import os
import importlib

# 自动导入当前目录下所有的 .py 文件
pkg_dir = os.path.dirname(__file__)
for filename in os.listdir(pkg_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = f".{filename[:-3]}"
        importlib.import_module(module_name, package=__package__)