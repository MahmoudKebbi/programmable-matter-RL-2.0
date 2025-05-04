import numpy as np
import json
import os


class ShapeTemplate:
    """Represents a pre-computed optimal transformation between two shapes."""

    def __init__(self, source_shape, target_shape, moves, name=""):
        self.source = source_shape
        self.target = target_shape
        self.moves = moves
        self.name = name

    def get_canonical_representation(self, shape):
        """Get a canonical representation of a shape for matching."""
        if not shape:
            return []

        # Normalize by translating to origin
        min_x = min(x for x, y in shape)
        min_y = min(y for x, y in shape)

        # Return normalized shape sorted for consistency
        return sorted((x - min_x, y - min_y) for x, y in shape)

    def matches_source(self, shape):
        """Check if a shape matches this template's source shape."""
        return self.get_canonical_representation(
            shape
        ) == self.get_canonical_representation(self.source)

    def matches_target(self, shape):
        """Check if a shape matches this template's target shape."""
        return self.get_canonical_representation(
            shape
        ) == self.get_canonical_representation(self.target)


class ShapeTemplateLibrary:
    """Manages a collection of shape transformation templates."""

    def __init__(self, library_file="shape_templates.json"):
        self.templates = []
        self.library_file = library_file
        self.load_templates()

    def add_template(self, template):
        """Add a new template to the library."""
        self.templates.append(template)

    def find_matching_template(self, source_shape, target_shape):
        """Find a template that matches the given source and target shapes."""
        canonical_source = ShapeTemplate([], [], []).get_canonical_representation(
            source_shape
        )
        canonical_target = ShapeTemplate([], [], []).get_canonical_representation(
            target_shape
        )

        for template in self.templates:
            if (
                template.get_canonical_representation(template.source)
                == canonical_source
                and template.get_canonical_representation(template.target)
                == canonical_target
            ):
                return template

        return None

    def add_from_successful_plan(
        self, source_shape, target_shape, plan, name="auto_generated"
    ):
        """Add a new template from a successful plan."""
        template = ShapeTemplate(source_shape, target_shape, plan, name)
        self.add_template(template)

        # Inform the user
        print(
            f"New shape template added: {len(source_shape)} blocks, {len(plan)} moves"
        )

        # Save immediately to avoid losing templates
        self.save_templates()

    def save_templates(self):
        """Save all templates to a JSON file."""
        data = []
        for template in self.templates:
            data.append(
                {
                    "name": template.name,
                    "source": template.source,
                    "target": template.target,
                    "moves": template.moves,
                }
            )

        with open(self.library_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_templates(self):
        """Load templates from a JSON file if it exists."""
        if os.path.exists(self.library_file):
            try:
                with open(self.library_file, "r") as f:
                    data = json.load(f)

                for item in data:
                    template = ShapeTemplate(
                        item["source"],
                        item["target"],
                        item["moves"],
                        item.get("name", ""),
                    )
                    self.templates.append(template)

                print(f"Loaded {len(self.templates)} shape templates")
            except Exception as e:
                print(f"Error loading templates: {e}")
