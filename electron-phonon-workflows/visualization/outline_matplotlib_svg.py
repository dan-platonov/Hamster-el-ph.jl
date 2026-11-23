"""Convert lettering in Matplotlib SVGs like the supplied figure to outlines.

Dependencies: matplotlib, lxml
Usage: python outline_matplotlib_svg.py input.svg [output.svg]

Tailored to this figure's DejaVu Sans text and explicit tspan coordinates.
This is not a general-purpose SVG converter. Labels become vector paths.
"""
from pathlib import Path
import argparse
import re

from lxml import etree as ET
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath, TextToPath


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=Path, help='Input Matplotlib SVG file')
parser.add_argument('output', nargs='?', type=Path, help='Output SVG (default: input_outlined.svg)')
args = parser.parse_args()
SOURCE = args.input
DEST = args.output or SOURCE.with_name(SOURCE.stem + '_outlined.svg')
if SOURCE.resolve() == DEST.resolve():
    parser.error('Choose a separate output file to preserve the original.')
NS = 'http://www.w3.org/2000/svg'
tree = ET.parse(str(SOURCE), ET.XMLParser(resolve_entities=False, no_network=True))
measure = TextToPath()


def style_of(node):
    return dict(item.strip().split(':', 1) for item in node.get('style', '').split(';') if ':' in item)


def outline(parent, text, style, x, y):
    font = style.get('font', '')
    size = float(re.search(r'([\d.]+)px', font).group(1))
    prop = FontProperties(family='DejaVu Sans', size=size,
                          style='oblique' if 'oblique' in font else 'normal')
    width = measure.get_text_width_height_descent(text, prop, False)[0]
    anchor = style.get('text-anchor', 'start').strip()
    x -= width * {'start': 0, 'middle': 0.5, 'end': 1}[anchor]
    path = TextPath((0, 0), text, size=size, prop=prop, usetex=False)
    commands = []
    letters = {MplPath.MOVETO: 'M', MplPath.LINETO: 'L',
               MplPath.CURVE3: 'Q', MplPath.CURVE4: 'C'}
    for points, code in path.iter_segments(curves=True, simplify=False):
        if code == MplPath.CLOSEPOLY:
            commands.append('z')
        elif code != MplPath.STOP:
            commands.append(letters[code] + ' '.join(f'{n:.6f}'.rstrip('0').rstrip('.') if n else '0'
                                                    for n in points))
    if commands:
        ET.SubElement(parent, f'{{{NS}}}path',
                      d=' '.join(commands),
                      transform=f'translate({x:.8f} {y:.8f}) scale(1 -1)',
                      style='fill: #000000; stroke: none')


texts = list(tree.findall(f'.//{{{NS}}}text'))
for text_node in texts:
    group = ET.Element(f'{{{NS}}}g')
    if text_node.get('transform'):
        group.set('transform', text_node.get('transform'))
    if text_node.get('id'):
        group.set('id', text_node.get('id'))
    description = ''.join(text_node.itertext()).strip()
    title = ET.SubElement(group, f'{{{NS}}}title')
    title.text = description
    if len(text_node):
        for span in text_node:
            assert span.tag == f'{{{NS}}}tspan'
            s = span.text or ''
            xs = [float(n) for n in span.get('x', '0').split()]
            ys = [float(n) for n in span.get('y', '0').split()]
            st = style_of(span)
            if len(xs) > 1:
                assert len(xs) == len(s)
                for index, char in enumerate(s):
                    if not char.isspace():
                        outline(group, char, st, xs[index], ys[min(index, len(ys) - 1)])
            else:
                outline(group, s, st, xs[0], ys[0])
    else:
        outline(group, text_node.text or '', style_of(text_node),
                float(text_node.get('x', 0)), float(text_node.get('y', 0)))
    group.tail = text_node.tail
    text_node.getparent().replace(text_node, group)

assert not tree.findall(f'.//{{{NS}}}text')
assert not tree.findall(f'.//{{{NS}}}tspan')
assert not tree.findall(f'.//{{{NS}}}image')
DEST.parent.mkdir(parents=True, exist_ok=True)
tree.write(str(DEST), encoding='utf-8', xml_declaration=True, pretty_print=True)
print(f'Converted {len(texts)} text elements to vector outlines: {DEST.name}')
