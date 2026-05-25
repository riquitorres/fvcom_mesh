"""
fvcom_mesh — Automatic unstructured mesh generator for FVCOM
============================================================
Based on the ADMESH+ methodology (Kang & Kubatko, GMD 2024,
https://doi.org/10.5194/gmd-17-1603-2024).

Typical usage
-------------
>>> from fvcom_mesh import MeshGenerator
>>> gen = MeshGenerator.from_config("config.yml")
>>> mesh = gen.run()
>>> mesh.write_fvcom("output/maldives")

Or from the command line::

    fvcom-mesh run config.yml
"""

from fvcom_mesh.core import MeshGenerator
from fvcom_mesh.mesh_quality import element_quality
from fvcom_mesh.dynamic_quality import DynamicQuality

__all__ = ["MeshGenerator", "element_quality", "DynamicQuality"]
__version__ = "0.1.0"
