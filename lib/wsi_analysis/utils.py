import xmltodict


def read_polygon_annotations_from_xml(xml_file_path):
    with open(xml_file_path, "r") as f:
        anns = xmltodict.parse(f.read())

    coordinates = anns["ASAP_Annotations"]["Annotations"]["Annotation"]
    if not isinstance(coordinates, list):
        coordinates = [coordinates]
    polygons = dict(Tumor=[], Exclusion=[])

    for coord in coordinates:
        if not coord["@Type"] == "Polygon":
            continue # Only polygons

        cs = coord["Coordinates"]["Coordinate"]
        polygon = [None]*len(cs)
        for c in cs:
            xy = (float(c["@X"]), float(c["@Y"]))
            polygon[int(c["@Order"])] = xy

        polygons[coord["@PartOfGroup"]].append(polygon)

    return polygons

