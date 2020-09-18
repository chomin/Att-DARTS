import sys
import argparse
from pathlib import Path

from graphviz import Digraph

import utils
import genotypes


def plot(genotype, bottleneck, filename, directory, view):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j, attn = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            # g.edge(u, v, label=f'{op}\n{attn}', fillcolor="gray")
            g.edge(u, v, label=f'{op} {attn}', fillcolor="gray")

    if not bottleneck:
        g.node("c_{k}", fillcolor='palegoldenrod')
        for i in range(steps):
            g.edge(str(i), "c_{k}", fillcolor="gray")
    else:
        g.node("c'_{k}", fillcolor='palegoldenrod')
        for i in range(steps):
            g.edge(str(i), "c'_{k}", fillcolor="gray")
        g.node("c_{k}", fillcolor='palegoldenrod')
        g.edge("c'_{k}", "c_{k}", label=bottleneck, fillcolor="gray")

    g.render(filename, directory=directory, view=view)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('visualize')
    parser.add_argument('genotype_name', type=str, help="genotype's name to visualize")
    parser.add_argument('--without_view', action='store_true', default=False, help='only save to pdf')
    parser.add_argument('--file_name', type=str, help='file name to save')
    parser.add_argument('--save_dir', type=Path, help='path to save pdf')
    args = parser.parse_args()

    genotype_name = args.genotype_name

    if genotype_name.endswith('.json'):
        genotype = utils.load_genotype(genotype_name)
    else:
        try:
            genotype = eval(f'genotypes.{genotype_name}')
        except AttributeError:
            print(f'{genotype_name} is not specified in genotypes.py')
            sys.exit(1)

    plot(genotype.normal, genotype.normal_bottleneck, f'normal_{genotype_name}', directory=args.save_dir,
         view=not args.without_view)
    plot(genotype.reduce, genotype.reduce_bottleneck, f'reduce_{genotype_name}', directory=args.save_dir,
         view=not args.without_view)

    print(f'plot genotype: {genotype}')
