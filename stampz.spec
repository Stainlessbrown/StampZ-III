# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all

# Version for PyInstaller spec - keep this in sync with __init__.py
VERSION = '3.0.2'

# Safely collect odfpy data files and imports
try:
    datas, binaries, hiddenimports = collect_all('odfpy')
except Exception:
    datas, binaries, hiddenimports = [], [], []

# Collect matplotlib data files and imports
try:
    matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all('matplotlib')
    datas += matplotlib_datas
    binaries += matplotlib_binaries
    hiddenimports += matplotlib_hiddenimports
except Exception:
    pass

# Collect openpyxl data files and imports
try:
    openpyxl_datas, openpyxl_binaries, openpyxl_hiddenimports = collect_all('openpyxl')
    datas += openpyxl_datas
    binaries += openpyxl_binaries
    hiddenimports += openpyxl_hiddenimports
except Exception:
    pass

# Collect seaborn data files and imports for Plot_3D
try:
    seaborn_datas, seaborn_binaries, seaborn_hiddenimports = collect_all('seaborn')
    datas += seaborn_datas
    binaries += seaborn_binaries
    hiddenimports += seaborn_hiddenimports
except Exception:
    pass

# Collect scikit-learn data files and imports for Plot_3D clustering and PCA
try:
    sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')
    datas += sklearn_datas
    binaries += sklearn_binaries
    hiddenimports += sklearn_hiddenimports
except Exception:
    pass

# Collect scipy data files and imports (sklearn dependency)
try:
    scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
    datas += scipy_datas
    binaries += scipy_binaries
    hiddenimports += scipy_hiddenimports
except Exception:
    pass

# Collect threadpoolctl (sklearn dependency)
try:
    threadpoolctl_datas, threadpoolctl_binaries, threadpoolctl_hiddenimports = collect_all('threadpoolctl')
    datas += threadpoolctl_datas
    binaries += threadpoolctl_binaries
    hiddenimports += threadpoolctl_hiddenimports
except Exception:
    pass

# Collect ezodf data files and imports for ODS file handling
try:
    ezodf_datas, ezodf_binaries, ezodf_hiddenimports = collect_all('ezodf')
    datas += ezodf_datas
    binaries += ezodf_binaries
    hiddenimports += ezodf_hiddenimports
except Exception:
    pass

# Collect lxml data files and imports for XML/HTML parsing
try:
    lxml_datas, lxml_binaries, lxml_hiddenimports = collect_all('lxml')
    datas += lxml_datas
    binaries += lxml_binaries
    hiddenimports += lxml_hiddenimports
except Exception:
    pass

# Collect tksheet data files and imports for Plot_3D spreadsheet interface
try:
    tksheet_datas, tksheet_binaries, tksheet_hiddenimports = collect_all('tksheet')
    datas += tksheet_datas
    binaries += tksheet_binaries
    hiddenimports += tksheet_hiddenimports
except Exception:
    pass

# Collect cv2 (OpenCV) data files and imports for perforation measurement
try:
    cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')
    datas += cv2_datas
    binaries += cv2_binaries
    hiddenimports += cv2_hiddenimports
except Exception:
    pass

# Add additional hidden imports
hiddenimports += [
    'initialize_env',  # Critical: Entry point environment setup
    'launch_selector',  # Main application launch selector
    'cv2',  # OpenCV for image processing
    'gui.gauge_perforation_ui',  # Perforation gauge measurement dialog
    'gui.perforation_ui',  # Legacy perforation measurement dialog
    'PIL.Image',
    'PIL.ImageTk',
    'PIL._tkinter_finder',
    'PIL._imaging',
    'PIL._imagingft',
    'PIL._imagingmath',
    'PIL._imagingmorph',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'PIL.ImageFilter',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.simpledialog',
    'tkinter.ttk',
    'tkinter.colorchooser',
    'tkinter.commondialog',
    '_tkinter',
    'numpy',
    'colorspacious',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',
    'pandas',
    'seaborn',
    'sklearn',
    'sklearn.cluster',
    'sklearn.cluster.KMeans',
    'sklearn.cluster._kmeans',
    'sklearn.decomposition',
    'sklearn.base',
    'sklearn.utils',
    'sklearn.utils._param_validation',
    'sklearn.utils.validation',
    'sklearn.utils.fixes',
    'sklearn.exceptions',
    'sklearn.metrics',
    'sklearn.metrics.pairwise',
    'scipy',
    'scipy.sparse',
    'scipy.linalg',
    'threadpoolctl',
    'odf.opendocument',
    'odf.table',
    'odf.text',
    'odf.style',
    'odf.number',
    'ezodf',
    'lxml',
    'lxml.etree',
    'lxml.html',
    'openpyxl',
    'tifffile',
    'tksheet',
]

# Platform specific settings
if sys.platform == 'darwin':
    # macOS - use original StampZ icon (convert on-the-fly if needed)
    icon_path = 'resources/StampZ.ico' if os.path.exists('resources/StampZ.ico') else None
    onefile = False  # Use --onedir for app bundles
elif sys.platform == 'win32':
    # Windows - use original StampZ icon
    icon_path = 'resources/StampZ.ico' if os.path.exists('resources/StampZ.ico') else None
    onefile = True
else:
    # Linux
    icon_path = None
    onefile = True

# Collect data files - only if they exist
if os.path.exists('resources'):
    datas += [('resources', 'resources')]
if os.path.exists('data'):
    datas += [('data', 'data')]

# Explicitly add templates directory to ensure Plot_3D templates are included
# This fixes the issue where Plot_3D export would fail in bundled apps due to missing template files
# The templates directory contains Plot3D_Template.ods which is required for the export functionality
if os.path.exists('data/templates'):
    datas += [('data/templates', 'data/templates')]

# Add Plot_3D configuration files (zoom presets, etc.)
if os.path.exists('plot3d/zoom_presets.json'):
    datas += [('plot3d/zoom_presets.json', 'plot3d')]

# Explicitly add critical modules that PyInstaller might miss
hiddenimports += ['initialize_env', 'launch_selector']

a = Analysis(
    ['main.py', 'initialize_env.py', 'launch_selector.py', 'gui/gauge_perforation_ui.py', 'gui/perforation_ui.py'],  # Explicitly include these modules
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.'],  # Look for hooks in current directory
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

if onefile:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='StampZ-III',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_path,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='StampZ-III',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_path,
    )
    
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='StampZ-III',
    )
    
    # macOS app bundle
    if sys.platform == 'darwin':
        app = BUNDLE(
            coll,
            name='StampZ-III.app',
            icon=icon_path,
            bundle_identifier='com.stainlessbrown.stampz-iii',
            version=VERSION,
        )
