import numpy as np
import time
from typing import ClassVar, Optional

from models import Path, Instruction

# TODO:
# - sort out the calculations for determining danger level
# - using event or processing time for previous instruction key vals?
# - include previous instructions in instruction calculations?

class PathAnalyser:
    _instance: ClassVar[Optional['PathAnalyser']] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Ensure only one instance of PathAnalyser exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the path analyser only once."""
        if not self._initialized:
            self._initialized = True
            self.paths: list[Path] = []
            
            # previous instructions are stored in a dictionary with the timestamp of the instruction
            # IDEALLY, we would also know how far the user has moved since the last instruction, but
            # I'm not sure how easy this will be to do just yet
            self.previous_instructions: dict[float, list[Instruction]] = []
            self.instructions: list[Instruction] = []

    def _analyse_path(self, path: Path) -> Instruction | None:
        """
        Analyse the properties of a path and determine if an instruction is necessary

        Args:
            path (Path): A path object

        Returns:
            Instruction: a single instruction
        """
        angle = path.angle
        length = path.length
        
        # determine a threshold for how much the angle needs to be in a certain length
        # we tell the user, if the path moves say 5 grids over 30 grids in length it is 
        # probably not necessary to inform the user but if it is more significant, we should
        
        # Determine if angle change is significant enough to warrant an instruction
        if length < (self.frame_height * 0.3):
            return None
            
        # Calculate danger based on angle and length
        if abs(angle) > 45:
            danger = "high"
        elif abs(angle) > 25:
            danger = "medium"
        else:
            danger = "low"
            
        # Determine instruction type based on angle change
        instruction_type = "bearing" if angle < 20 else "curve" if angle < 35 else "turn"
        direction = "straight" if path.start.x == path.end.x else "left" if path.start.x > path.end.x else "right"
            
        return Instruction(
            direction=direction,
            danger=danger,
            distance=length,
            angle_change=angle,
            length=length,
            instruction_type=instruction_type
        )
        
    def _analyse_corners(self, path: Path) -> list[Instruction]:
        """
        Analyse the properties of the corner and determine an instruction for the user

        Args:
            path (Path): A path object

        Returns:
            list[Instruction]: A list of instructions for the application to understand
        """
        instructions = []
        
        for corner in path.corners:
            # Higher y value means closer to bottom of frame/user
            distance = corner.start.y
            
            # TODO: determine whether to keep this logic?
            # Calculate position-based weight for angle importance
            # At bottom of frame (max y), weight is 0.3
            # At top of frame (min y), weight is 0.7
            # angle_weight = 0.3 + (0.4 * (1 - distance / self.frame_height))
            # height_weight = 1 - angle_weight
            
            # Use exponential functions for height and angle multipliers
            height_multiplier = np.exp((np.log(2) / self.frame_height) * distance) - 1
            # absolute value of angle change to prevent negative values
            angle_multiplier = np.exp((np.log(2) / 90) * abs(corner.angle_change)) - 1 
                        
            # TODO: determine whether to keep this logic?
            # danger_value = height_multiplier * height_weight + angle_multiplier * angle_weight
            danger_value = (height_multiplier * 0.7) + (angle_multiplier * 0.3)
            
            # TODO: determine whether to keep this logic?
            # # Near bottom of frame, only consider very sharp angles
            # if distance > (self.frame_height * 0.7) and abs(corner.angle_change) < 30:
            #     danger_value *= 0.5  # Reduce danger for shallow angles near bottom
            
            match danger_value:
                case value if value > 0.75:
                    danger = "immediate"
                case value if value > 0.65:
                    danger = "high"
                case value if value > 0.45:
                    danger = "medium"
                case _:
                    danger = "low"
        
            # Create instruction
            instruction = Instruction(
                direction=corner.direction,
                danger=danger,
                distance=distance,
                angle_change=corner.angle_change,
                length=corner.length,
                instruction_type="turn" if corner.sharpness == "sharp" else "curve"
            )
            
            instructions.append(instruction)
            
        return instructions
        
    def _analyse_instructions(self, instructions: list[Instruction]) -> list[Instruction]:
        """
        Analyse the instructions and determine if some are redundant
        
        Args:
            instructions (list[Instruction]): A list of instructions

        Returns:
            list[Instruction]: A list of instructions for the application to understand
        """
        return instructions
    
    def _analyse_previous_instructions(
        self, 
        previous_instructions: list[Instruction], 
        current_instructions: list[Instruction]
    ) -> list[Instruction]:
        """
        Analyse the previous instructions and use it to enrich the current instruction.

        Args:
            previous_instructions (list[Instruction])
            current_instruction (list[Instruction])

        Returns:
            list[Instruction]: An updated list of instructions, enriched by previous instructions
        """
        return current_instructions

    def __call__(self, frame_height: int, paths: list[Path]) -> list[Path]:
        """
        Process and analyse the paths.
        
        Args:
            frame_height (int): The height of the frame
            paths (list[Path]): A list of paths

        Returns:
            list[Instruction]: A list of instructions for the application to understand
        """
        self.paths = paths
        self.frame_height = frame_height
        
        self.instructions = []
        
        # TODO: determine if we want event or processing time - if event we need to take from phone
        processing_time = time.time()
        
        for path in self.paths:
            path_instruction = self._analyse_path(path)
            if path_instruction: self.instructions.append(path_instruction)
            
            if path.corners:
                corner_instructions = self._analyse_corners(path)
                self.instructions.extend(corner_instructions)
                
        # Remove instructions that are redundant
        self.instructions = self._analyse_instructions(self.instructions)
        
        # Sort first by instruction type (bearing last), then by danger level
        def sort_key(instruction):
            type_order = {'turn': 0, 'curve': 0, 'bearing': 1}
            danger_order = {'immediate': 0, 'high': 1, 'medium': 2, 'low': 3}
            
            return (
                type_order[instruction.instruction_type],
                danger_order[instruction.danger]
            )
            
        self.instructions = sorted(self.instructions, key=sort_key)
        
        self.instructions = self._analyse_previous_instructions(self.previous_instructions, self.instructions)
        
        # TODO: process this in relation to previous instructions?
        # say, if we have three frames in a row where the instruction is turn right and the 
        # danger is increasing, we should definitely instruct the user to move
        self.previous_instructions[processing_time] = self.instructions
        
        # Remove any previous instructions that are more than 5s before the current processing time
        # For two reasons:
        #   1. want to keep the size as small as possible
        #   2. no point keeping out of date instructions
        # Only issue I can see here is if the user is stood still but it would be hard to determine that
        self.previous_instructions = {
            timestamp: instructions 
            for timestamp, instructions in self.previous_instructions.items()
            if processing_time - timestamp <= 5
        }
        
        return self.instructions


# Create singleton for export
path_analyser = PathAnalyser()