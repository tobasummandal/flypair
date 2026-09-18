import io
import struct

import numpy as np

from flypair.video import (
    _complete_bilateral_part,
    _mesh_part_outlines,
    _mesh_to_polygon,
    _parse_ascii_stl,
    _parse_stl,
)


def test_parse_ascii_stl_and_project_polygon():
    stl = '''
solid test_fly
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid
'''

    for stream in (io.StringIO(stl), io.BytesIO(stl.encode())):
        mesh = _parse_ascii_stl(stream)
        assert mesh.shape == (1, 3, 3)

    unit = _mesh_to_polygon(mesh, heading_deg=45.0, scale=1.0)
    poly = _mesh_to_polygon(mesh, heading_deg=45.0, scale=10.0)
    assert poly.shape[0] >= 3
    assert poly.shape[1] == 2
    assert np.allclose(np.linalg.norm(poly, axis=1), 10 * np.linalg.norm(unit, axis=1))


def test_parse_binary_stl():
    header = b"binary test".ljust(80, b"\0")
    normal = (0.0, 0.0, 1.0)
    vertices = (0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    data = header + struct.pack("<I12fH", 1, *normal, *vertices, 0)

    mesh = _parse_stl(data)

    assert mesh.shape == (1, 3, 3)
    assert np.allclose(mesh[0], [[0, 0, 0], [2, 0, 0], [0, 1, 0]])


def test_left_only_parts_are_mirrored():
    mesh = np.array([[[0, 1, 0], [1, 2, 0], [2, 3, 0]]], dtype=float)

    complete = _complete_bilateral_part(mesh, "l_wing.stl")

    assert complete.shape == (2, 3, 3)
    assert np.allclose(complete[1, :, 1], [-1, -2, -3])
    assert _complete_bilateral_part(mesh, "c_head.stl") is mesh


def test_part_outlines_keep_components_in_common_coordinates():
    left = np.array([[[0, 1, 0], [1, 1, 0], [0, 2, 0]]], dtype=float)
    right = left.copy()
    right[:, :, 1] *= -1

    outlines = _mesh_part_outlines((left, right))

    assert len(outlines) == 2
    assert outlines[0][:, 1].mean() > 0
    assert outlines[1][:, 1].mean() < 0
