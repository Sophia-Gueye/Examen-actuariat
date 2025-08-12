#!/usr/bin/env python3
"""
Script pour corriger les imports parse_version dans tous les fichiers Python
"""

import os
import re
from pathlib import Path

def fix_parse_version_imports(file_path):
    """Corrige les imports parse_version dans un fichier"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern pour détecter l'import problématique
    old_import = re.compile(r'from sklearn\.utils import.*parse_version.*')
    
    # Si le pattern est trouvé, on le remplace
    if old_import.search(content):
        print(f"Correction de {file_path}")
        
        # Remplacer l'import problématique
        new_import = """try:
    from packaging import version
    parse_version = version.parse
except ImportError:
    try:
        from pkg_resources import parse_version
    except ImportError:
        from distutils.version import LooseVersion as parse_version"""
        
        content = old_import.sub(new_import, content)
        
        # Écrire le fichier corrigé
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    return False

def main():
    """Corrige tous les fichiers Python dans src/examen_actuariat/"""
    src_dir = Path('src/examen_actuariat')
    
    if not src_dir.exists():
        print(f"Répertoire {src_dir} non trouvé")
        return
    
    files_fixed = 0
    for py_file in src_dir.glob('*.py'):
        if fix_parse_version_imports(py_file):
            files_fixed += 1
    
    print(f"\n{files_fixed} fichier(s) corrigé(s)")
    print("Vous pouvez maintenant tester les imports avec:")
    print("poetry run python scripts/train.py")

if __name__ == "__main__":
    main()
