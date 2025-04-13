import numpy as np
import gym
from gym import spaces
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class EnhancedCurriculumManager:
    """Enhanced curriculum manager for programmable matter with smoother transitions"""

    def __init__(self, initial_grid_size=12, auto_adjust=True):
        self.n = initial_grid_size
        self.m = initial_grid_size
        self.auto_adjust = auto_adjust

        # Current curriculum state
        self.level = 0
        self.sub_level = 0
        self.phase = "1_block_short"
        self.block_count = 1
        self.source_shape = "single"
        self.target_shape = "single"
        self.current_distance = 2
        self.current_obstacles = []
        self.difficulty_factor = 1.0

        # Progress tracking
        self.consecutive_successes = 0
        self.required_successes = 3
        self.stuck_episodes = 0
        self.max_stuck_episodes = 200

        # Micro-curriculum state (for difficult levels)
        self.micro_level = None
        self.micro_step = 0
        self.micro_attempts = 0

        # Block shape definitions
        self.shapes_by_count = {
            1: ["single"],
            2: ["line2_h", "line2_v"],
            3: ["line3_h", "line3_v", "L3_ul", "L3_ur", "L3_dl", "L3_dr"],
            4: [
                "square2x2",
                "line4_h",
                "line4_v",
                "L4_ul",
                "L4_ur",
                "L4_dl",
                "L4_dr",
                "T4_u",
                "T4_d",
                "T4_l",
                "T4_r",
            ],
            5: ["plus", "line5_h", "line5_v"],
            6: ["U_shape", "H_shape"],
            7: ["T7_shape"],
            8: ["square3x3_hollow"],
            9: ["square3x3"],
        }

        # Define curriculum levels with sub-levels
        # Each level has multiple sub-levels for smoother progression
        self.curriculum = self._create_enhanced_curriculum()
        self.max_level = max(level for level, _ in self.curriculum.keys())

        # Create initial environment
        self.env = self.create_environment()

    def _create_enhanced_curriculum(self):
        """Create enhanced curriculum with sub-levels for smoother progression"""
        curriculum = {}

        # Level 0: Single block, short distance
        curriculum[(0, 0)] = {
            "phase": "1_block_short",
            "block_count": 1,
            "source_shape": "single",
            "target_shape": "single",
            "distance": 2,
            "obstacles": 0,
            "required_successes": 3,
        }

        # Level 1: Single block, medium distance
        curriculum[(1, 0)] = {
            "phase": "1_block_medium",
            "block_count": 1,
            "source_shape": "single",
            "target_shape": "single",
            "distance": 4,
            "obstacles": 0,
            "required_successes": 3,
        }

        # Level 2: Single block with few obstacles
        curriculum[(2, 0)] = {
            "phase": "1_block_obstacles",
            "block_count": 1,
            "source_shape": "single",
            "target_shape": "single",
            "distance": 4,
            "obstacles": 2,
            "required_successes": 3,
        }

        # Level 3: Two blocks, horizontal line, no change in shape
        curriculum[(3, 0)] = {
            "phase": "2_blocks_same_shape",
            "block_count": 2,
            "source_shape": "line2_h",
            "target_shape": "line2_h",
            "distance": 3,
            "obstacles": 0,
            "required_successes": 3,
        }

        # Level 3, sub-level 1: Two blocks with small obstacle
        curriculum[(3, 1)] = {
            "phase": "2_blocks_obstacles",
            "block_count": 2,
            "source_shape": "line2_h",
            "target_shape": "line2_h",
            "distance": 3,
            "obstacles": 1,
            "required_successes": 3,
        }

        # Level 4: Two blocks, shape change (horizontal to vertical)
        curriculum[(4, 0)] = {
            "phase": "2_blocks_shape_change",
            "block_count": 2,
            "source_shape": "line2_h",
            "target_shape": "line2_v",
            "distance": 3,
            "obstacles": 0,
            "required_successes": 3,
        }

        # Level 4, sub-level 1: Two blocks, shape change with obstacle
        curriculum[(4, 1)] = {
            "phase": "2_blocks_shape_change_obstacles",
            "block_count": 2,
            "source_shape": "line2_h",
            "target_shape": "line2_v",
            "distance": 3,
            "obstacles": 2,
            "required_successes": 3,
        }

        # Level 5: Three blocks line, no shape change
        curriculum[(5, 0)] = {
            "phase": "3_blocks_same_shape",
            "block_count": 3,
            "source_shape": "line3_h",
            "target_shape": "line3_h",
            "distance": 3,
            "obstacles": 0,
            "required_successes": 4,
        }

        # Level 5, sub-level 1: Three blocks line with small obstacle
        curriculum[(5, 1)] = {
            "phase": "3_blocks_obstacles",
            "block_count": 3,
            "source_shape": "line3_h",
            "target_shape": "line3_h",
            "distance": 3,
            "obstacles": 1,
            "required_successes": 4,
        }

        # === CRITICAL TRANSITION ZONE - Breaking up level 6 ===

        # Level 5, sub-level 2: Three blocks, very simple shape change (horizontal to vertical)
        curriculum[(5, 2)] = {
            "phase": "3_blocks_simple_shape_change",
            "block_count": 3,
            "source_shape": "line3_h",
            "target_shape": "line3_v",
            "distance": 2,  # Shorter distance to make it easier
            "obstacles": 0,
            "required_successes": 4,
        }

        # Level 5, sub-level 3: Three blocks, L shape (simpler shape change)
        curriculum[(5, 3)] = {
            "phase": "3_blocks_L_shape",
            "block_count": 3,
            "source_shape": "line3_h",
            "target_shape": "L3_ul",  # L shape
            "distance": 3,
            "obstacles": 0,
            "required_successes": 4,
        }

        # Level 5, sub-level 4: Three blocks, L shape (simpler shape change) with obstacle
        curriculum[(5, 4)] = {
            "phase": "3_blocks_L_shape_obstacles",
            "block_count": 3,
            "source_shape": "line3_h",
            "target_shape": "L3_ul",  # L shape
            "distance": 3,
            "obstacles": 1,
            "required_successes": 4,
        }

        # Level 6: Three blocks, more complex shape change
        curriculum[(6, 0)] = {
            "phase": "3_blocks_complex_shapes",
            "block_count": 3,
            "source_shape": "L3_ul",
            "target_shape": "L3_dr",  # Different L shape orientation
            "distance": 3,
            "obstacles": 0,
            "required_successes": 5,
        }

        # Level 6, sub-level 1: Three blocks, complex shape change with obstacle
        curriculum[(6, 1)] = {
            "phase": "3_blocks_complex_obstacles",
            "block_count": 3,
            "source_shape": "L3_ul",
            "target_shape": "L3_dr",
            "distance": 4,
            "obstacles": 2,
            "required_successes": 5,
        }

        # Level 7: Four blocks, simple shape
        curriculum[(7, 0)] = {
            "phase": "4_blocks_line",
            "block_count": 4,
            "source_shape": "line4_h",
            "target_shape": "line4_h",
            "distance": 3,
            "obstacles": 0,
            "required_successes": 5,
        }

        # Level 7, sub-level 1: Four blocks, simple shape change
        curriculum[(7, 1)] = {
            "phase": "4_blocks_simple_change",
            "block_count": 4,
            "source_shape": "line4_h",
            "target_shape": "line4_v",
            "distance": 3,
            "obstacles": 0,
            "required_successes": 5,
        }

        # Level 8: Four blocks, complex shape change
        curriculum[(8, 0)] = {
            "phase": "4_blocks_complex_change",
            "block_count": 4,
            "source_shape": "line4_h",
            "target_shape": "square2x2",
            "distance": 3,
            "obstacles": 0,
            "required_successes": 6,
        }

        # Level 8, sub-level 1: Four blocks, complex shape with obstacles
        curriculum[(8, 1)] = {
            "phase": "4_blocks_complex_obstacles",
            "block_count": 4,
            "source_shape": "square2x2",
            "target_shape": "T4_u",
            "distance": 4,
            "obstacles": 2,
            "required_successes": 6,
        }

        # Level 9: More complex shapes and transformations
        curriculum[(9, 0)] = {
            "phase": "advanced_shapes",
            "block_count": 5,
            "source_shape": "plus",
            "target_shape": "line5_h",
            "distance": 4,
            "obstacles": 2,
            "required_successes": 6,
        }

        # Level 10: Final challenge
        curriculum[(10, 0)] = {
            "phase": "final_challenge",
            "block_count": 6,
            "source_shape": "U_shape",
            "target_shape": "H_shape",
            "distance": 5,
            "obstacles": 3,
            "required_successes": 7,
        }

        return curriculum

    def get_environment(self):
        """Get current environment"""
        return self.env

    def get_level(self):
        """Get current level"""
        return self.level

    def get_phase(self):
        """Get current phase"""
        return self.phase

    def create_environment(self):
        """Create environment with current settings"""
        # Here, you would integrate with your actual environment
        # This is a placeholder assuming your environment takes these parameters
        env = self._create_env()
        return env

    def _create_env(self):
        """Create environment based on current parameters

        Replace with actual environment instantiation based on your implementation
        """
        # Add your environment creation code here
        # This is placeholder code - you'll need to adapt to your actual environment class
        from environment import ProgrammableMatterEnv

        env = ProgrammableMatterEnv(
            n=self.n,
            m=self.m,
            block_count=self.block_count,
            source_shape=self.source_shape,
            target_shape=self.target_shape,
            distance=self.current_distance,
            obstacles=self.current_obstacles,
        )

        return env

    def evaluate_progress(self, episode_reward, info):
        """Evaluate progress and update curriculum if needed"""
        # Check if level was solved
        solved = info.get("solved", False) or episode_reward > 90

        if solved:
            self.consecutive_successes += 1
            self.stuck_episodes = 0
            print(f"🎉 Success! {self.consecutive_successes}/{self.required_successes}")

            # Check if we should advance to next level or sub-level
            if self.consecutive_successes >= self.required_successes:
                # Reset success counter
                self.consecutive_successes = 0

                # Handle micro-curriculum completion
                if self.micro_level is not None:
                    self._advance_micro_level()
                    return True

                # Advance in main curriculum
                return self._advance_level()
        else:
            self.consecutive_successes = 0
            self.stuck_episodes += 1

            if self.stuck_episodes % 50 == 0:
                print(f"⚠️ Stuck for {self.stuck_episodes} episodes...")

            # Check if stuck too long
            if self.stuck_episodes >= self.max_stuck_episodes:
                # If we're in a micro-level, try next step or skip
                if self.micro_level is not None:
                    if self.micro_step < 3:
                        self._advance_micro_level()
                    else:
                        self.emergency_skip_micro_level()
                    return True

                # If normal level is too difficult, create micro-curriculum
                if self.auto_adjust and self.level >= 5:
                    print(
                        f"🔍 Creating micro-curriculum for level {self.level}.{self.sub_level}"
                    )
                    self._create_micro_curriculum()
                    return True

                # If auto-adjust is disabled or level is too low, skip to next level
                if not self.auto_adjust and self.level > 0:
                    print(
                        f"⏭️ Skipping level {self.level}.{self.sub_level} after {self.stuck_episodes} failed attempts"
                    )
                    return self._advance_level(skip=True)

        return False

    def _advance_level(self, skip=False):
        """Advance to the next level or sub-level"""
        # Determine the next level and sub-level
        next_level = self.level
        next_sub_level = self.sub_level + 1

        # Check if the next sub-level exists
        if (next_level, next_sub_level) not in self.curriculum:
            # If not, go to the next main level, sub-level 0
            next_level += 1
            next_sub_level = 0

        # Check if we've completed the curriculum
        if next_level > self.max_level:
            print("🏆 Curriculum completed! Staying at max level.")
            next_level = self.max_level
            next_sub_level = 0

        # Update level and sub-level
        self.level = next_level
        self.sub_level = next_sub_level

        # Update environment parameters from curriculum
        self._update_params_from_curriculum()

        # Create new environment
        self.env = self.create_environment()

        # Print progress message
        if skip:
            print(f"⏭️ Skipped to level {self.level}.{self.sub_level} ({self.phase})")
        else:
            print(f"⬆️ Advanced to level {self.level}.{self.sub_level} ({self.phase})")

        return True

    def _update_params_from_curriculum(self):
        """Update parameters from curriculum definition"""
        level_info = self.curriculum.get((self.level, self.sub_level))

        if level_info:
            self.phase = level_info["phase"]
            self.block_count = level_info["block_count"]
            self.source_shape = level_info["source_shape"]
            self.target_shape = level_info["target_shape"]
            self.current_distance = level_info["distance"]

            # Handle obstacles
            if isinstance(level_info["obstacles"], int):
                count = level_info["obstacles"]
                # Generate random obstacles - the environment will place them properly
                self.current_obstacles = count
            else:
                self.current_obstacles = level_info["obstacles"]

            # Update required successes
            self.required_successes = level_info.get(
                "required_successes", self.required_successes
            )

    def _create_micro_curriculum(self):
        """Create micro-curriculum for current level"""
        self.micro_level = (self.level, self.sub_level)
        self.micro_step = 0
        self.micro_attempts = 0

        # Store original parameters
        self.original_params = {
            "block_count": self.block_count,
            "source_shape": self.source_shape,
            "target_shape": self.target_shape,
            "distance": self.current_distance,
            "obstacles": self.current_obstacles,
        }

        # Simplify the level
        # 1. Reduce distance
        self.current_distance = max(2, self.current_distance - 1)

        # 2. Remove obstacles
        self.current_obstacles = 0

        # 3. Simplify shape change if possible
        if self.source_shape != self.target_shape:
            # Try to find a simpler shape transformation
            block_count = self.block_count
            available_shapes = self.shapes_by_count.get(block_count, ["single"])

            # If multiple shapes are available, use current source shape
            # but find a simpler target shape to transition to
            if len(available_shapes) > 1 and self.source_shape in available_shapes:
                # Find a shape that's a simpler transformation
                # For now, just use the first available shape
                self.target_shape = available_shapes[0]

        # Create new environment with simplified parameters
        self.env = self.create_environment()

        print(
            f"🔍 Created micro-level: blocks={self.block_count}, "
            f"shapes={self.source_shape}->{self.target_shape}, "
            f"distance={self.current_distance}, obstacles={self.current_obstacles}"
        )

    def _advance_micro_level(self):
        """Advance to next step in micro-curriculum"""
        self.micro_step += 1
        self.micro_attempts = 0

        if self.micro_step == 1:
            # Step 1: Keep simplified shapes but restore original distance
            self.current_distance = self.original_params["distance"]

        elif self.micro_step == 2:
            # Step 2: Add some obstacles (half of original)
            if isinstance(self.original_params["obstacles"], int):
                self.current_obstacles = max(1, self.original_params["obstacles"] // 2)
            else:
                # If obstacles were specified as positions
                if len(self.original_params["obstacles"]) > 0:
                    self.current_obstacles = self.original_params["obstacles"][
                        :1
                    ]  # Just one obstacle
                else:
                    self.current_obstacles = []

        elif self.micro_step == 3:
            # Step 3: Restore original target shape
            self.target_shape = self.original_params["target_shape"]

        else:
            # Final step: Return to original level parameters
            self.block_count = self.original_params["block_count"]
            self.source_shape = self.original_params["source_shape"]
            self.target_shape = self.original_params["target_shape"]
            self.current_distance = self.original_params["distance"]
            self.current_obstacles = self.original_params["obstacles"]

            # Exit micro-curriculum
            self.micro_level = None
            print("✅ Completed micro-curriculum, returning to regular level")

        # Create new environment with updated parameters
        self.env = self.create_environment()

        if self.micro_level is not None:
            print(
                f"🔍 Advanced micro-level step {self.micro_step}: "
                f"shapes={self.source_shape}->{self.target_shape}, "
                f"distance={self.current_distance}, obstacles={self.current_obstacles}"
            )

        return True

    def emergency_skip_micro_level(self):
        """Emergency skip micro-level after too many failed attempts"""
        if self.micro_level is None:
            return

        print(
            f"⚠️ Emergency: Skipping micro-curriculum after {self.micro_attempts} failed attempts"
        )

        # Return to original level parameters
        self.block_count = self.original_params["block_count"]
        self.source_shape = self.original_params["source_shape"]
        self.target_shape = self.original_params["target_shape"]
        self.current_distance = self.original_params["distance"]
        self.current_obstacles = self.original_params["obstacles"]

        # Skip to next level
        self.micro_level = None
        self._advance_level(skip=True)
