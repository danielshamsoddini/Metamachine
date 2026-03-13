# Random Robot Generation API

This document describes the random robot generation API in MetaMachine.
The API provides a generic core layer that can be extended by factory-specific plugins.

## Architecture

```
metamachine/robot_factory/random_generation.py  (CORE - Abstract Layer)
├── DockGender, DockSpec, ComponentDockRegistry  (Generic dock system)
├── ComponentBudget                              (Generic budget base class)
├── GenerationConfig, GenerationMode             (Generic config)
├── BaseRandomGraphGenerator                     (Abstract generator base)
├── RandomGenerationCapability                   (Factory mixin)
└── visualize_graph, graph_to_yaml_dict          (Utilities)
```

## Quick Start

### Using the Factory Pattern

```python
from metamachine.robot_factory import load_plugins_from, get_robot_factory

# Load your plugin package
load_plugins_from("metamachine_plugins/private_plugins")
factory = get_robot_factory("my_factory")

# Create a budget understood by the selected factory
budget = factory.create_default_budget()

# Generate random robot using factory
robot = factory.generate_random_robot(budget=budget)
robot.render()
```

## Core API Reference

### DockSpec (Generic)

```python
from metamachine.robot_factory import DockSpec, DockGender

# Create dock specifications
male_dock = DockSpec.male("0m", position=0)
female_dock = DockSpec.female("0f", position=0)

# Parse from string
dock = DockSpec.from_string("1f")  # Female dock at position 1
```

### ComponentDockRegistry (Generic)

```python
from metamachine.robot_factory import ComponentDockRegistry, DockSpec, DockGender
from metamachine.robot_factory.morphology import ComponentType

# Create custom registry for a new robot type
registry = ComponentDockRegistry()
registry.register_component(
    ComponentType.BALL,
    [DockSpec.female("0f"), DockSpec.female("1f")]
)

# Check if docks can connect
can_connect = registry.can_connect(male_dock, female_dock)  # True
```

### ComponentBudget (Generic Base)

```python
from metamachine.robot_factory import ComponentBudget
from metamachine.robot_factory.morphology import ComponentType

# Create generic budget
budget = ComponentBudget.create(
    components={
        ComponentType.BALL: 3,
        ComponentType.STICK4: 6,
    },
    min_legs=2,
)
```

### RandomGenerationCapability (Factory Mixin)

```python
from metamachine.robot_factory import (
    BaseRobotFactory,
    RandomGenerationCapability,
    ComponentDockRegistry,
    BaseRandomGraphGenerator,
)

class MyFactory(BaseRobotFactory, RandomGenerationCapability):
    def get_dock_registry(self) -> ComponentDockRegistry:
        return MY_DOCK_REGISTRY
    
    def create_random_generator(self, budget, config) -> BaseRandomGraphGenerator:
        return MyRandomGraphGenerator(budget, self.get_dock_registry(), config)
```

## Utility Functions

### Converting Graph to YAML

```python
from metamachine.robot_factory import graph_to_yaml_dict

graph = generator.generate()
yaml_dict = graph_to_yaml_dict(graph)

# Save to file
import yaml
with open("robot_config.yaml", "w") as f:
    yaml.dump(yaml_dict, f)
```

### Visualizing Graph Structure

```python
from metamachine.robot_factory import visualize_graph

# Generate DOT format for visualization
dot_str = visualize_graph(graph, output_path="robot_graph.dot")

# Render with graphviz (if installed)
# dot -Tpng robot_graph.dot -o robot_graph.png
```

## Examples

### Generate Random Modular Legs Robots

```bash
# See examples/good_robot_distribution.py for a complete example
python examples/good_robot_distribution.py
```

## Creating a New Robot Factory with Random Generation

1. **Create your dock registry**:
```python
MY_DOCK_REGISTRY = ComponentDockRegistry()
MY_DOCK_REGISTRY.register_component(MyComponentType.JOINT, [...])
```

2. **Create your budget class** (optional, for convenience):
```python
class MyBudget(ComponentBudget):
    @classmethod
    def for_default(cls):
        return cls(components={...}, min_legs=2)
```

3. **Create your generator**:
```python
class MyRandomGraphGenerator(BaseRandomGraphGenerator):
    def generate(self) -> RobotGraph:
        # Your generation logic
        pass
```

4. **Implement the factory mixin**:
```python
class MyFactory(BaseRobotFactory, RandomGenerationCapability):
    def get_dock_registry(self):
        return MY_DOCK_REGISTRY
    
    def create_random_generator(self, budget, config):
        return MyRandomGraphGenerator(budget, self.get_dock_registry(), config)
```
