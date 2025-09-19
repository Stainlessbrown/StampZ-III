# Analysis Manager Refactoring Plan

## Current Problem
- `analysis_manager.py`: **2559 lines** - too large for maintenance
- Multiple responsibilities mixed together
- Hard to find specific functionality
- Risk of bugs when modifying

## Proposed Structure

### 1. Core Analysis Manager (analysis_manager.py) - ~300 lines
- Main coordinator class
- Imports and delegates to specialized managers
- Maintains backward compatibility

### 2. Color Analysis Manager (color_analysis_manager.py) - ~800 lines
- Color library management
- Sample comparison
- Spectral analysis
- Color data export

### 3. Data Export Manager (data_export_manager.py) - ~600 lines
- All export functionality (ODS, Plot3D, reports)
- Export formatting and file handling
- Unified data logging integration

### 4. Black Ink Manager (black_ink_manager.py) - ~400 lines
- Black ink extraction dialog and logic
- Image processing coordination

### 5. Measurement Manager (measurement_manager.py) - ~300 lines
- Precision measurements coordination
- Measurement data handling

### 6. Database Manager (database_manager.py) - ~400 lines
- Spreadsheet viewing
- Database operations
- Data persistence

## Benefits
- **Easier maintenance** - find code faster
- **Reduced complexity** - each file has single responsibility  
- **Better testing** - test individual components
- **Team development** - multiple developers can work simultaneously
- **Reduced merge conflicts** - changes isolated to relevant files

## Migration Strategy
1. **Extract managers one by one** - start with least dependencies
2. **Maintain interface compatibility** - no changes to calling code
3. **Test each extraction** - ensure nothing breaks
4. **Update imports gradually** - clean up as we go

## Implementation Priority
1. Data Export Manager (most isolated)
2. Black Ink Manager (self-contained)
3. Database Manager (clear boundaries)
4. Color Analysis Manager (largest chunk)
5. Measurement Manager (newest code)
6. Core Analysis Manager (coordinator)