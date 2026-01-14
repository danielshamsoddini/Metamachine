#!/usr/bin/env python3
"""
MuJoCo XML Model Inspector

Investigate MuJoCo XML files for sim-to-real gap analysis.
Extracts and reports key physical properties including:
- Mass distribution (total mass, per-body mass, inertia)
- Geometry properties (sizes, positions, collision properties)
- Joint and actuator specifications
- Material and friction parameters
- Simulation settings (timestep, solver, etc.)

Usage:
    python scripts/investigate_mujoco_xml.py path/to/robot.xml
    python scripts/investigate_mujoco_xml.py path/to/robot.xml --detailed
    python scripts/investigate_mujoco_xml.py path/to/robot.xml --export report.txt
    python scripts/investigate_mujoco_xml.py path/to/robot.xml --compare path/to/other.xml

For typical tripod robot:
    python scripts/investigate_mujoco_xml.py metamachine/assets/robots/nominal_legotripod2.xml

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mujoco
    from lxml import etree
except ImportError as e:
    print(f"ERROR: Missing dependencies: {e}")
    print("Please install: pip install mujoco lxml")
    sys.exit(1)


class MuJoCoInspector:
    """Inspect and analyze MuJoCo XML model files."""
    
    def __init__(self, xml_path: str):
        """
        Initialize inspector with XML file.
        
        Args:
            xml_path: Path to MuJoCo XML file
        """
        self.xml_path = Path(xml_path).resolve()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        
        # Parse XML
        self.tree = etree.parse(str(self.xml_path))
        self.root = self.tree.getroot()
        
        # Load MuJoCo model
        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self.data = mujoco.MjData(self.model)
            self.mujoco_loaded = True
        except Exception as e:
            print(f"WARNING: Could not load MuJoCo model: {e}")
            print("Will provide XML-only analysis.")
            self.model = None
            self.data = None
            self.mujoco_loaded = False
    
    def print_section_header(self, title: str, char: str = "="):
        """Print a formatted section header."""
        width = 80
        print(f"\n{char * width}")
        print(f"{title.center(width)}")
        print(f"{char * width}\n")
    
    def get_mass_summary(self) -> Dict:
        """Get mass distribution summary."""
        summary = {}
        
        if self.mujoco_loaded:
            # Total mass from compiled model
            summary['total_mass'] = float(np.sum(self.model.body_mass))
            summary['body_masses'] = {}
            
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    mass = float(self.model.body_mass[i])
                    if mass > 1e-6:  # Only show bodies with meaningful mass
                        summary['body_masses'][body_name] = mass
        
        # Also parse from XML for comparison
        xml_masses = {}
        total_xml_mass = 0.0
        
        for geom in self.root.findall('.//geom[@mass]'):
            name = geom.get('name', 'unnamed')
            mass = float(geom.get('mass'))
            xml_masses[name] = mass
            total_xml_mass += mass
        
        summary['xml_total_mass'] = total_xml_mass
        summary['xml_geom_masses'] = xml_masses
        
        return summary
    
    def get_inertia_info(self) -> Dict:
        """Get inertia tensor information."""
        info = {}
        
        if self.mujoco_loaded:
            info['body_inertias'] = {}
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and self.model.body_mass[i] > 1e-6:
                    # MuJoCo stores inertia as 3x3 matrix
                    inertia = self.model.body_inertia[i].copy()
                    info['body_inertias'][body_name] = {
                        'diagonal': inertia.tolist(),
                        'mass': float(self.model.body_mass[i])
                    }
        
        return info
    
    def get_geometry_info(self) -> Dict:
        """Get geometry properties."""
        info = {
            'geoms': [],
            'total_geoms': 0
        }
        
        if self.mujoco_loaded:
            for i in range(self.model.ngeom):
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                geom_type = self.model.geom_type[i]
                geom_size = self.model.geom_size[i].copy()
                
                type_names = {
                    mujoco.mjtGeom.mjGEOM_PLANE: 'plane',
                    mujoco.mjtGeom.mjGEOM_SPHERE: 'sphere',
                    mujoco.mjtGeom.mjGEOM_CYLINDER: 'cylinder',
                    mujoco.mjtGeom.mjGEOM_BOX: 'box',
                    mujoco.mjtGeom.mjGEOM_CAPSULE: 'capsule',
                    mujoco.mjtGeom.mjGEOM_MESH: 'mesh',
                }
                
                info['geoms'].append({
                    'name': geom_name or f'geom_{i}',
                    'type': type_names.get(geom_type, f'type_{geom_type}'),
                    'size': geom_size.tolist(),
                    'friction': self.model.geom_friction[i].tolist(),
                })
            
            info['total_geoms'] = self.model.ngeom
        
        return info
    
    def get_joint_actuator_info(self) -> Dict:
        """Get joint and actuator specifications."""
        info = {
            'joints': [],
            'actuators': [],
            'total_joints': 0,
            'total_actuators': 0,
            'dof': 0
        }
        
        if self.mujoco_loaded:
            info['total_joints'] = self.model.njnt
            info['total_actuators'] = self.model.nu
            info['dof'] = self.model.nv
            
            # Joint info
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                joint_type = self.model.jnt_type[i]
                
                type_names = {
                    mujoco.mjtJoint.mjJNT_FREE: 'free',
                    mujoco.mjtJoint.mjJNT_BALL: 'ball',
                    mujoco.mjtJoint.mjJNT_SLIDE: 'slide',
                    mujoco.mjtJoint.mjJNT_HINGE: 'hinge',
                }
                
                joint_info = {
                    'name': joint_name or f'joint_{i}',
                    'type': type_names.get(joint_type, f'type_{joint_type}'),
                }
                
                # Range for limited joints
                if self.model.jnt_limited[i]:
                    joint_info['range'] = self.model.jnt_range[i].tolist()
                
                # Damping and armature
                if self.model.dof_damping[i] > 0:
                    joint_info['damping'] = float(self.model.dof_damping[i])
                if self.model.dof_armature[i] > 0:
                    joint_info['armature'] = float(self.model.dof_armature[i])
                
                info['joints'].append(joint_info)
            
            # Actuator info
            for i in range(self.model.nu):
                act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                
                act_info = {
                    'name': act_name or f'actuator_{i}',
                    'ctrl_range': self.model.actuator_ctrlrange[i].tolist(),
                    'force_range': self.model.actuator_forcerange[i].tolist(),
                    'gear': self.model.actuator_gear[i].tolist(),
                }
                
                info['actuators'].append(act_info)
        
        return info
    
    def get_simulation_settings(self) -> Dict:
        """Get simulation configuration settings."""
        settings = {}
        
        # Parse from XML
        option = self.root.find('.//option')
        if option is not None:
            settings['integrator'] = option.get('integrator', 'Euler')
            settings['timestep'] = float(option.get('timestep', 0.002))
            
        compiler = self.root.find('.//compiler')
        if compiler is not None:
            settings['angle_unit'] = compiler.get('angle', 'radian')
            settings['coordinate'] = compiler.get('coordinate', 'global')
            settings['inertiafromgeom'] = compiler.get('inertiafromgeom', 'false')
        
        if self.mujoco_loaded:
            settings['timestep_compiled'] = float(self.model.opt.timestep)
            settings['iterations'] = int(self.model.opt.iterations)
            settings['nconmax'] = int(self.model.nconmax)
            settings['njmax'] = int(self.model.njmax)
        
        return settings
    
    def get_contact_friction_settings(self) -> Dict:
        """Get contact and friction parameters."""
        settings = {
            'default_friction': None,
            'geom_frictions': {}
        }
        
        # Default friction from XML
        default_geom = self.root.find('.//default/geom')
        if default_geom is not None:
            friction_str = default_geom.get('friction')
            if friction_str:
                settings['default_friction'] = [float(x) for x in friction_str.split()]
        
        # Per-geom friction
        for geom in self.root.findall('.//geom[@friction]'):
            name = geom.get('name', 'unnamed')
            friction_str = geom.get('friction')
            settings['geom_frictions'][name] = [float(x) for x in friction_str.split()]
        
        if self.mujoco_loaded:
            # Compiled friction settings
            settings['solref'] = self.model.opt.o_solref.tolist()
            settings['solimp'] = self.model.opt.o_solimp.tolist()
        
        return settings
    
    def print_full_report(self, detailed: bool = False):
        """Print comprehensive analysis report."""
        print("\n" + "=" * 80)
        print(f"MuJoCo Model Analysis: {self.xml_path.name}".center(80))
        print("=" * 80)
        print(f"File path: {self.xml_path}")
        
        # Mass Summary
        self.print_section_header("MASS DISTRIBUTION")
        mass_info = self.get_mass_summary()
        
        if self.mujoco_loaded:
            print(f"Total Robot Mass (compiled): {mass_info['total_mass']:.4f} kg")
            print(f"\nMass by Body:")
            for body_name, mass in sorted(mass_info['body_masses'].items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"  {body_name:20s}: {mass:8.4f} kg ({mass/mass_info['total_mass']*100:5.1f}%)")
        
        print(f"\nTotal Geom Mass (from XML): {mass_info['xml_total_mass']:.4f} kg")
        if detailed:
            print(f"\nMass by Geom:")
            for geom_name, mass in sorted(mass_info['xml_geom_masses'].items(),
                                         key=lambda x: x[1], reverse=True):
                print(f"  {geom_name:20s}: {mass:8.4f} kg")
        
        # Inertia
        if self.mujoco_loaded and detailed:
            self.print_section_header("INERTIA TENSORS")
            inertia_info = self.get_inertia_info()
            for body_name, data in inertia_info['body_inertias'].items():
                print(f"{body_name}:")
                print(f"  Mass: {data['mass']:.4f} kg")
                print(f"  Inertia (diagonal): [{', '.join(f'{x:.6f}' for x in data['diagonal'])}]")
        
        # Geometry
        self.print_section_header("GEOMETRY")
        geom_info = self.get_geometry_info()
        print(f"Total Geometries: {geom_info['total_geoms']}")
        
        if detailed and geom_info['geoms']:
            print("\nGeometry Details:")
            for geom in geom_info['geoms'][:20]:  # Limit to first 20
                print(f"  {geom['name']:20s} ({geom['type']:8s}): size={geom['size']}, friction={geom['friction']}")
            if len(geom_info['geoms']) > 20:
                print(f"  ... and {len(geom_info['geoms']) - 20} more")
        
        # Joints and Actuators
        self.print_section_header("JOINTS & ACTUATORS")
        joint_info = self.get_joint_actuator_info()
        print(f"Degrees of Freedom: {joint_info['dof']}")
        print(f"Total Joints: {joint_info['total_joints']}")
        print(f"Total Actuators: {joint_info['total_actuators']}")
        
        if joint_info['joints']:
            print("\nActuated Joints:")
            for joint in joint_info['joints']:
                if joint['type'] not in ['free']:  # Skip free joints
                    info_str = f"  {joint['name']:20s} ({joint['type']:8s})"
                    if 'range' in joint:
                        info_str += f" range={joint['range']}"
                    if 'damping' in joint:
                        info_str += f" damping={joint['damping']:.3f}"
                    if 'armature' in joint:
                        info_str += f" armature={joint['armature']:.3f}"
                    print(info_str)
        
        if detailed and joint_info['actuators']:
            print("\nActuator Details:")
            for act in joint_info['actuators']:
                print(f"  {act['name']:20s}: ctrl_range={act['ctrl_range']}, gear={act['gear'][:2]}")
        
        # Simulation Settings
        self.print_section_header("SIMULATION SETTINGS")
        sim_settings = self.get_simulation_settings()
        for key, value in sim_settings.items():
            print(f"  {key:25s}: {value}")
        
        # Contact/Friction
        self.print_section_header("CONTACT & FRICTION")
        contact_info = self.get_contact_friction_settings()
        
        if contact_info['default_friction']:
            print(f"Default Friction: {contact_info['default_friction']}")
        
        if detailed and contact_info['geom_frictions']:
            print("\nPer-Geom Friction:")
            for name, friction in contact_info['geom_frictions'].items():
                print(f"  {name:20s}: {friction}")
        
        if self.mujoco_loaded:
            print(f"\nSolver Parameters:")
            print(f"  solref (stiffness, damping): {contact_info['solref']}")
            print(f"  solimp (impedance params): {contact_info['solimp']}")
        
        print("\n" + "=" * 80)
    
    def export_report(self, output_path: str):
        """Export analysis to text file."""
        import sys
        from io import StringIO
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            self.print_full_report(detailed=True)
            report = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report exported to: {output_path}")
    
    def compare_with(self, other_xml: str):
        """Compare this model with another XML file."""
        print(f"\nComparing {self.xml_path.name} with {Path(other_xml).name}...")
        
        other = MuJoCoInspector(other_xml)
        
        # Compare masses
        self.print_section_header("MASS COMPARISON")
        mass1 = self.get_mass_summary()
        mass2 = other.get_mass_summary()
        
        total1 = mass1.get('total_mass', mass1.get('xml_total_mass', 0))
        total2 = mass2.get('total_mass', mass2.get('xml_total_mass', 0))
        
        print(f"Model 1 ({self.xml_path.name}): {total1:.4f} kg")
        print(f"Model 2 ({Path(other_xml).name}): {total2:.4f} kg")
        print(f"Difference: {abs(total1 - total2):.4f} kg ({abs(total1-total2)/total1*100:.1f}%)")
        
        # Compare DOF and actuators
        self.print_section_header("DOF & ACTUATOR COMPARISON")
        joint1 = self.get_joint_actuator_info()
        joint2 = other.get_joint_actuator_info()
        
        print(f"DOF:       {joint1['dof']:3d} vs {joint2['dof']:3d}")
        print(f"Joints:    {joint1['total_joints']:3d} vs {joint2['total_joints']:3d}")
        print(f"Actuators: {joint1['total_actuators']:3d} vs {joint2['total_actuators']:3d}")


def main():
    parser = argparse.ArgumentParser(
        description='Investigate MuJoCo XML files for sim-to-real analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python scripts/investigate_mujoco_xml.py metamachine/assets/robots/nominal_legotripod2.xml
  
  # Detailed report with all information
  python scripts/investigate_mujoco_xml.py robot.xml --detailed
  
  # Export report to file
  python scripts/investigate_mujoco_xml.py robot.xml --export report.txt
  
  # Compare two models
  python scripts/investigate_mujoco_xml.py robot1.xml --compare robot2.xml
        """
    )
    
    parser.add_argument('xml_file', type=str,
                       help='Path to MuJoCo XML file')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed analysis including all components')
    parser.add_argument('--export', '-e', type=str, metavar='OUTPUT',
                       help='Export report to text file')
    parser.add_argument('--compare', '-c', type=str, metavar='XML2',
                       help='Compare with another XML file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.xml_file):
        # Try relative to metamachine assets
        alt_path = os.path.join(PROJECT_ROOT, 'metamachine', 'assets', 'robots', args.xml_file)
        if os.path.exists(alt_path):
            args.xml_file = alt_path
        else:
            print(f"ERROR: File not found: {args.xml_file}")
            sys.exit(1)
    
    try:
        # Create inspector
        inspector = MuJoCoInspector(args.xml_file)
        
        # Run analysis
        if args.compare:
            inspector.print_full_report(detailed=args.detailed)
            inspector.compare_with(args.compare)
        elif args.export:
            inspector.export_report(args.export)
            print(f"\nTo view: cat {args.export}")
        else:
            inspector.print_full_report(detailed=args.detailed)
        
        # Print summary tips
        print("\n" + "=" * 80)
        print("SIM-TO-REAL GAP ANALYSIS TIPS")
        print("=" * 80)
        print("""
Key things to check for sim-to-real transfer:
  1. Total robot mass - should match physical robot ±5%
  2. Joint damping and armature - affects responsiveness
  3. Friction coefficients - critical for contact dynamics
  4. Timestep - smaller = more accurate but slower
  5. Solver iterations - affects constraint satisfaction
  6. Actuator force limits - should match real motor specs
  7. Inertia distribution - affects angular dynamics

To measure physical robot:
  - Weigh individual components
  - Measure dimensions with calipers
  - Test motor torque limits
  - Observe surface friction behavior
  - Tune timestep/iterations for stability vs speed
        """)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
