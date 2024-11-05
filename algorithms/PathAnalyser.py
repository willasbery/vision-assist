import cv2
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Optional

from config import grid_size
from corner_detection import detect_corner


@dataclass
class Corner:
    """Represents a detected corner in a path."""
    type: str  # 'left' or 'right'
    confidence: float
    angle_change: float  # Total angle change through the corner

@dataclass
class Path:
    """
    Represents a path through a grid with its properties and sections.
    Each path can have far, mid, and close sections which are also Path objects.
    """
    grids: list[dict]  # List of grid dictionaries in the path
    total_cost: float  # Total cost/distance of the path
    section_type: Optional[str] = None  # One of: 'far', 'mid', 'close', or None for full path
    
    def __post_init__(self):
        """Initialize section paths if this is a full path."""
        self._far_section: Optional[Path] = None
        self._mid_section: Optional[Path] = None
        self._close_section: Optional[Path] = None
        self._corner: Optional[Corner] = None
        
        # Only calculate sections for full paths (non-section paths)
        if not self.section_type and self.grids:
            self._calculate_sections()
            self._detect_corner()
    
    @property
    def start(self) -> tuple[int, int]:
        """Get the starting coordinates of the path."""
        if not self.grids:
            return (0, 0)
        start_grid = self.grids[0]
        return (start_grid['coords']['x'], start_grid['coords']['y'])
    
    @property
    def end(self) -> tuple[int, int]:
        """Get the ending coordinates of the path."""
        if not self.grids:
            return (0, 0)
        end_grid = self.grids[-1]
        return (end_grid['coords']['x'], end_grid['coords']['y'])
    
    @property
    def grid_coords(self) -> list[tuple[int, int]]:
        """Get list of grid coordinates in the path."""
        return [(grid['coords']['x'], grid['coords']['y']) for grid in self.grids]
    
    @property
    def length(self) -> float:
        """Calculate the length of the path."""
        return np.hypot(self.end[0] - self.start[0], 
                       self.end[1] - self.start[1])
    
    @property
    def angle(self) -> float:
        """Calculate the angle of the path in degrees."""
        delta_x = self.end[0] - self.start[0]
        delta_y = self.end[1] - self.start[1]
        angle = np.arctan2(delta_y, delta_x)
        return np.degrees(angle)
    
    @property
    def has_corner(self) -> bool:
        """Check if the path has a detected corner."""
        return self._corner is not None
    
    @property
    def corner(self) -> Optional[Corner]:
        """Get the detected corner information if it exists."""
        return self._corner
    
    def _normalize_slope(self, slope: float) -> float:
        """Normalize infinite slopes to a large finite value."""
        return 1000 if abs(slope) == float('inf') else slope
    
    def _detect_corner(self) -> None:
        """
        Detect if there's a corner in the path based on section slopes.
        Updates the self._corner property with the detection result.
        """
        if not self.has_all_sections:
            return
        
        # Get and normalize slopes
        close_slope = self._normalize_slope(self.close_section.angle)
        mid_slope = self._normalize_slope(self.mid_section.angle)
        far_slope = self._normalize_slope(self.far_section.angle)
        
        # Calculate slope differences
        diff_close_mid = abs(close_slope - mid_slope)
        diff_mid_far = abs(mid_slope - far_slope)
        total_angle_change = abs(far_slope - close_slope)
        max_slope = max(abs(close_slope), abs(mid_slope), abs(far_slope))
        
        # Calculate confidence based on both angle differences
        confidence = min(diff_close_mid, diff_mid_far) / max_slope if max_slope > 0 else 0
        
        # Detect corner type
        if abs(far_slope) > abs(mid_slope) > abs(close_slope):
            self._corner = Corner('left', confidence, total_angle_change)
        elif abs(close_slope) > abs(mid_slope) > abs(far_slope):
            self._corner = Corner('right', confidence, total_angle_change)
    
    @property
    def has_all_sections(self) -> bool:
        """Check if the path has grids in all sections."""
        if self.section_type:
            raise AttributeError("Section paths don't have sub-sections")
        return bool(self._far_section and self._far_section.grids and 
                   self._mid_section and self._mid_section.grids and 
                   self._close_section and self._close_section.grids)
    
    def _calculate_sections(self) -> None:
        """
        Calculate far, mid, and close sections of the path.
        Each section is created as its own Path object.
        """
        if not self.grids:
            return
            
        y_coords = [grid['coords']['y'] for grid in self.grids]
        min_y = min(y_coords)
        max_y = max(y_coords)
        y_range = abs(max_y - min_y)
        
        far_threshold = min_y + (y_range * 0.1)  # <10% from the top
        mid_threshold = min_y + (y_range * 0.4)  # >10% and <40% from the top
        
        # Create section paths
        far_grids = [grid for grid in self.grids if grid['coords']['y'] <= far_threshold]
        self._far_section = Path(
            grids=far_grids,
            total_cost=self.total_cost * (len(far_grids) / len(self.grids)),
            section_type='far'
        )
        
        mid_grids = [grid for grid in self.grids 
                    if far_threshold < grid['coords']['y'] <= mid_threshold]
        self._mid_section = Path(
            grids=mid_grids,
            total_cost=self.total_cost * (len(mid_grids) / len(self.grids)),
            section_type='mid'
        )
        
        close_grids = [grid for grid in self.grids if grid['coords']['y'] > mid_threshold]
        self._close_section = Path(
            grids=close_grids,
            total_cost=self.total_cost * (len(close_grids) / len(self.grids)),
            section_type='close'
        )
    
    @property
    def far_section(self) -> 'Path':
        """Get the far section of the path."""
        if self.section_type:
            raise AttributeError("Section paths don't have sub-sections")
        return self._far_section or Path([], float('inf'), 'far')
    
    @property
    def mid_section(self) -> 'Path':
        """Get the mid section of the path."""
        if self.section_type:
            raise AttributeError("Section paths don't have sub-sections")
        return self._mid_section or Path([], float('inf'), 'mid')
    
    @property
    def close_section(self) -> 'Path':
        """Get the close section of the path."""
        if self.section_type:
            raise AttributeError("Section paths don't have sub-sections")
        return self._close_section or Path([], float('inf'), 'close')


@dataclass
class PathColors:
    """Represents color variants for path visualization."""
    close: tuple[int, int, int]
    mid: tuple[int, int, int]
    far: tuple[int, int, int]


class PathAnalyser:
    _instance: ClassVar[Optional['PathAnalyser']] = None
    _initialized: bool = False
    
    PATH_COLORS = [
        PathColors((0, 255, 0), (0, 200, 0), (0, 150, 0)),  # Green variants
        PathColors((255, 0, 0), (200, 0, 0), (150, 0, 0)),  # Red variants
        PathColors((0, 0, 255), (0, 0, 200), (0, 0, 150)),  # Blue variants
        PathColors((255, 255, 0), (200, 200, 0), (150, 150, 0))  # Yellow variants
    ]
    
    def __new__(cls):
        """Ensure only one instance of PathAnalyser exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the path detector only once."""
        if not self._initialized:
            self._initialized = True
            self.frame: Optional[np.ndarray] = None
            self.paths: list[Path] = []
            
    def _draw_path_grid(self, grid: dict, color: tuple[int, int, int]) -> None:
        """Draw a single grid on the frame."""
        x, y = grid['coords']['x'], grid['coords']['y']
        
        grid_corners = np.array([
            [x, y],
            [x + grid_size, y],
            [x + grid_size, y + grid_size],
            [x, y + grid_size]
        ], np.int32)
        
        cv2.fillPoly(self.frame, [grid_corners], color)
    
    def _draw_path_sections(self, path: Path, path_colors: PathColors, path_idx: int) -> None:
        """Draw path sections with measurements and corner detection."""
        for grid in path.far_section.grids:
            self._draw_path_grid(grid, path_colors.far)
        for grid in path.mid_section.grids:
            self._draw_path_grid(grid, path_colors.mid)
        for grid in path.close_section.grids:
            self._draw_path_grid(grid, path_colors.close)
        
        if path.has_all_sections:
            # Draw connecting lines
            for section in [path.far_section, path.mid_section, path.close_section]:
                cv2.line(self.frame, section.start, section.end, (255, 255, 255), 2)
            
            # Add measurements text
            text_offset = path_idx * 25
            for section, name in [
                (path.far_section, "Far"),
                (path.mid_section, "Mid"),
                (path.close_section, "Close")
            ]:
                cv2.putText(
                    self.frame,
                    f"Path {path_idx + 1} {name}: {section.angle:.2f}, {section.length:.2f}",
                    (section.start[0], section.start[1] + text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )
            
            if path.has_corner:
                cv2.putText(
                    self.frame,
                    f"Path {path_idx + 1}: {path.corner.type.capitalize()} corner detected: {path.corner.confidence:.2f}",
                    (30, 320 + path_idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )
            
    def __call__(self, frame: np.ndarray, paths: list[Path]) -> np.ndarray:
        """Process the frame and visualize all paths."""
        self.frame = frame
        self.paths = paths
        
        for idx, path in enumerate(self.paths):
            path_colors = self.PATH_COLORS[idx % len(self.PATH_COLORS)]
            self._draw_path_sections(path, path_colors, idx)
                
        return self.frame


# Create singleton for export
path_analyser = PathAnalyser()