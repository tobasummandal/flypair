import io

from flypair.video import _mesh_to_polygon, _parse_ascii_stl


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

    mesh = _parse_ascii_stl(io.StringIO(stl))
    assert mesh.shape == (1, 3, 3)

    poly = _mesh_to_polygon(mesh, heading_deg=45.0, scale=10.0)
    assert poly.shape[0] >= 3
    assert poly.shape[1] == 2
