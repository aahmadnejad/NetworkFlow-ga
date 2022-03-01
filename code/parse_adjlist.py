from typing import List
from pathlib import Path
from argparse import Namespace, ArgumentParser
import pickle as pkl

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from ga import GA

def is_degenerated(x: list) -> bool:
    return False if len(x) % 2 == 0 else True


def build_node_idx(n_idx: int, sp_node: dict) -> int:
    return n_idx - 1 if n_idx not in list(sp_node.keys()) else sp_node[n_idx]


def read_txt_file(p: Path) -> List[List[int]]:
    lines = []
    with open(str(p), 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = [int(l) for l in line.split(" ") if l != ""]
            lines.append(line)
    return lines


def main(arguments: Namespace) -> None:
    f_path = Path(arguments.in_file)
    if not f_path.is_file():
        raise FileNotFoundError(f"The selected file did not found on {f_path.parent}")

    save_at = Path(arguments.save_at) if arguments.save_at is not None else None
    if save_at is not None:
        save_at.mkdir(parents=True, exist_ok=True)

    lines = read_txt_file(f_path)

    n_nodes = lines[0][0]

    source_idx = 0
    sink_idx = n_nodes - 1
    special_nodes_idx = {0: source_idx, -1: sink_idx}

    capacities = np.zeros((n_nodes, n_nodes), dtype=np.int32)
    all_edges = []

    for line in lines[1:]:
        n_idx = build_node_idx(line[0], special_nodes_idx)
        n_edges = line[2:]
        if is_degenerated(n_edges):
            raise Exception("this formatted file is degenerated!!")

        edges = []
        for idx in range(0, len(n_edges), 2):
            e_idx = build_node_idx(n_edges[idx], special_nodes_idx)
            e_cap = n_edges[idx + 1]
            edges.append((n_idx, e_idx, e_cap))
            capacities[n_idx, e_idx] = e_cap

        all_edges += edges
        
    #########################################
    if arguments.generations:
        res = GA(all_edges, n_nodes, arguments.in_gens, arguments.generations).solution()
    else:
        res = GA(all_edges, n_nodes, arguments.in_gens).solution()
    if arguments.p:
        graph = nx.Graph()
        labels = {}
        for s, e, c in res:
            graph.add_edge(s, e)
            labels[(s, e)] = c

        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        if save_at is None:
            plt.show()
        else:
            s_p = save_at.joinpath("test_graph.png")
            plt.savefig(s_p)
            print(f"graph saved at {str(s_p)}")

    if save_at is not None:
        s_p = save_at.joinpath("capacities.pkl")
        with open(str(s_p), "wb") as f:
            pkl.dump(capacities.tolist(), file=f)

        print(f"capacity matrix saved at {str(s_p)}")

        s_p = save_at.joinpath("edges.pkl")
        with open(str(s_p), "wb") as f:
            pkl.dump(res, file=f)

        print(f"All edges saved at {str(s_p)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', dest="in_file", help="input .txt file", type=str)
    parser.add_argument('--print', dest="p", help="print graph", action="store_true")
    parser.add_argument('--ingens', dest="in_gens", help="initial generation", type=int)
    parser.add_argument('--generations', dest="generations", help="number of generations", type=int)
    parser.add_argument("--save_at", help="save in determined path", type=str, required=False)
    args: Namespace = parser.parse_args()
    main(arguments=args)
