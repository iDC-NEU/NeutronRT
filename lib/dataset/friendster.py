import os
import dgl
import numpy as np
import torch as th

from dgl.transforms import reorder_graph
from dgl.data.utils import (
    download,
    extract_archive,
    generate_mask_tensor,
    load_graphs,
    save_graphs,
)

from torch import Tensor
from dgl import DGLGraph
from dgl.data.dgl_dataset import DGLDataset

from .utils import NUM_CLASSES, NUM_FEATURES, dummy_feat, dummy_label


class FriendsterDataset(DGLDataset):
    def __init__(
            self,
            raw_dir=None,
            hash_key=(),
            force_reload=False,
            verbose=False,
            transform=None
    ):
        # https://snap.stanford.edu/data/com-Friendster.html
        name = 'friendster'
        url = 'https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz'
        
        super(FriendsterDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            save_dir=None,
            hash_key=hash_key,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        txt = np.loadtxt(
            os.path.join(
                self.raw_path, "com-friendster.ungraph.txt"
            )
        )
        edge_index = th.from_numpy(txt)
        del txt

        # reindex
        _, mapping = edge_index.transpose_(0, 1).reshape(-1).unique(return_inverse=True)
        edge_index: Tensor = mapping.reshape(2, -1).contiguous()
        del mapping

        num_nodes = 6560_8366

        u, v = edge_index
        self._graph: DGLGraph = dgl.graph(
            data=(u, v),
            num_nodes=num_nodes,
        )
        # self._graph = dgl.to_bidirected(self._graph)

        features = dummy_feat(num_nodes, NUM_FEATURES)
        labels = dummy_label(num_nodes, NUM_CLASSES)

        self._graph.ndata["feat"] = features
        self._graph.ndata["label"] = labels

        self._graph = reorder_graph(
            self._graph,
            node_permute_algo="rcmk",
            edge_permute_algo="dst",
            store_ids=False,
        )


    def has_cache(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        if os.path.exists(graph_path):
            return True
        return False


    def save(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        save_graphs(graph_path, self._graph)


    def load(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self._print_info()


    def download(self):
        if self.url is not None:
            dl_file_path = os.path.join(self.raw_path, os.path.basename(self.url))
            download(self.url, path=dl_file_path)
            extract_archive(dl_file_path, self.raw_path)


    def _print_info(self):
        if self.verbose:
            print("Finished data loading.")
            print("  NumNodes: {}".format(self._graph.num_nodes()))
            print("  NumEdges: {}".format(self._graph.num_edges()))
            print("  NumFeats: {}".format(self._graph.ndata["feat"].shape[1]))
            print("  NumClasses: {}".format(self.num_classes))


    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return NUM_CLASSES

    def __getitem__(self, idx):
        assert idx == 0, "Twitter Dataset only has one graph"
        if self._transform is None:
            return self._graph
        else:
            return self._transform(self._graph)

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return 1