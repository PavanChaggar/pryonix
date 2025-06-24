import numpy as np
from lxml import etree

def get_node_attributes(graph):
    n_nodes = 83
    coords = np.empty((n_nodes, 3), dtype=float)
    nID = np.empty(n_nodes, dtype=int)
    region = np.empty(n_nodes, dtype=object)
    labels = np.empty(n_nodes, dtype=object)
    lobe = np.empty(n_nodes, dtype=object)
    hemisphere = np.empty(n_nodes, dtype=object)

    for i in range(n_nodes):
        for j in graph[i]:
            key = j.attrib.get("key")
            if key == "d1":
                coords[i, 0] = float(j.text)
            elif key == "d2":
                coords[i, 1] = float(j.text)
            elif key == "d3":
                coords[i, 2] = float(j.text)
            elif key == "d4":
                nID[i] = int(j.text)
            elif key == "d5":
                region[i] = j.text
            elif key == "d6":
                labels[i] = j.text
            elif key == "d7":
                lobe[i] = j.text
            elif key == "d8":
                hemisphere[i] = j.text

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return nID, labels, region, lobe, hemisphere, x, y, z

def get_adjacency_matrix(graph, N):
    L = np.zeros((N, N))
    N_matrix = np.zeros((N, N))

    for edge in graph[N:]:
        i = int(edge.attrib["source"])
        j = int(edge.attrib["target"])

        for child in edge:
            if child.attrib.get("key") == "d9":
                n = float(child.text)
            elif child.attrib.get("key") == "d10":
                l = float(child.text)

        N_matrix[i-1, j-1] = n
        L[i-1, j-1] = l

    return np.tril(N_matrix) + np.tril(N_matrix, -1).T, np.tril(L) + np.tril(L, -1).T


def load_graphml(graph_path):
    tree = etree.parse(graph_path)
    root = tree.getroot()
    ces = list(root)

    parc = get_node_attributes(ces[-1])

    N, L = get_adjacency_matrix(ces[-1], len(parc[1]))
    return parc, N, L


def load_parcellation(graph_path):
    tree = etree.parse(graph_path)
    root = tree.getroot()
    ces = list(root)
    
    return get_node_attributes(ces[-1])


for_dict = {1: "node", 2: "node", 3: "node", 4: "node", 5: "node", 6: "node", 7: "node", 8: "node", 9: "edge", 10: "edge"}
type_dict = {1: "double", 2: "double", 3: "double", 4: "string", 5: "string", 6: "string", 7: "string", 8: "string", 9: "int", 10: "double"}
name_dict = {1: "dn_position_x", 2: "dn_position_y", 3: "dn_position_z", 4: "dn_correspondence_id", 5: "dn_region", 6: "fn_fsname", 7: "dn_lobe", 8: "dn_hemisphere", 9: "number_of_fibers", 10: "fiber_length"}
field_dict = {1: "x", 2: "y", 3: "z", 4: "ID", 5: "Cortex", 6: "Label", 7: "Lobe", 8: "Hemisphere"}


def add_keys(root):
    for i in range(1, 11):
        c = etree.SubElement(root, "key")
        c.attrib["attr.name"] = name_dict[i]
        c.attrib["attr.type"] = type_dict[i]
        c.attrib["for"] = for_dict[i]
        c.attrib["id"] = f"d{i}"


def make_xml():
    xdoc = etree.ElementTree(etree.Element("graphml"))
    root = xdoc.getroot()
    add_keys(root)

    return xdoc


def add_nodes(connectome, c):
    for j in range(len(connectome.parc)):
        g = etree.SubElement(c, "node")
        g.attrib["id"] = str(j + 1)

        for i in range(1, 9):
            d = etree.SubElement(g, "data")
            d.attrib["key"] = f"d{i}"
            d.text = str(getattr(connectome.parc[j], field_dict[i]))


def add_edges(connectome, c):
    n_edges = np.argwhere(connectome.n_matrix > 0)

    for edge in n_edges:
        s, t = edge[0]
        g = etree.SubElement(c, "edge")
        g.attrib["source"] = str(s)
        g.attrib["target"] = str(t)
        
        d = etree.SubElement(g, "data")
        d.attrib["key"] = "d9"
        d.text = str(connectome.n_matrix[s, t])

        d = etree.SubElement(g, "data")
        d.attrib["key"] = "d10"
        d.text = str(connectome.l_matrix[s, t])


def save_connectome(filename, connectome):
    xdoc = make_xml()
    r = xdoc.getroot()
    c = etree.SubElement(r, "graph")
    add_nodes(connectome, c)
    add_edges(connectome, c)
    xdoc.write(filename, pretty_print=True)
