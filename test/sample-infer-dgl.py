import argparse
import os
import time
import logging

import dgl
import dgl.nn as dglnn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

# set logger file
logging.basicConfig(
    filename='logs/dgl.log',
    filemode='a+',
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        # self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        logging.debug("4")
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l != len(self.layers) - 1:
                #h = F.relu(h)
                h = self.dropout(h)
            h = layer(block, h)
        return h

    def updates_initialization(self, g, device, use_uva, updated_edge_size, snapshots):        
        total_edges = g.num_edges()
        # Generate random indices for sampling edges
        perm = torch.randperm(total_edges)
        sampled_indices = perm[:updated_edge_size*snapshots]
        logging.debug(f"Total edges: {total_edges}")
        logging.debug(f"Sampled indices: {sampled_indices}")
        # Split sampled indices into 10 equal parts
        split_size = sampled_indices.shape[0] // snapshots
        split_indices = torch.split(sampled_indices, split_size)
        split_indices = torch.stack(list(split_indices))
        return split_indices
        # Get source and destination nodes for each batch of edge
    def affect_layer_detection(self, g, device, use_uva, split_indice_batch, snapshots):
        # Get total number of edges 
        batch = split_indice_batch
        # Get source and destination nodes for the sampled edges
        src, dst = g.edges()
        batch_dst = dst[batch.cpu()].unique().to(device)
        #logging.debug("Destination nodes:", batch_dst)
        #batch_src = src[batch.cpu()].unique().to(device)
        current_dst=batch_dst
        sampler = MultiLayerFullNeighborSampler(1)
        #logging.debug("has {} affected nodes".format(len(batch_dst)))
        for hop in range(len(self.layers)-1):
            #logging.debug(f"Processing layer {hop}")
            dataloader = DataLoader(
                g,
                current_dst,
                sampler, 
                device=device,
                batch_size=10000,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva
            )
            src_nodes = []
            for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                current_dst=input_nodes
        return current_dst     

    def affect_layer_cache_detection(self, g, device, use_uva, split_indice_batch, snapshots):
        # Get total number of edges 
        batch = split_indice_batch
        # Get source and destination nodes for the sampled edges
        src, dst = g.edges()
        batch_dst = dst[batch.cpu()].unique().to(device)
        #logging.debug("Destination nodes:", batch_dst)
        #batch_src = src[batch.cpu()].unique().to(device)
        current_dst=batch_dst
        sampler = MultiLayerFullNeighborSampler(1)
        #logging.debug("has {} affected nodes".format(len(batch_dst)))
        for hop in range(len(self.layers)-1):
            #logging.debug(f"Processing layer {hop}")
            dataloader = DataLoader(
                g,
                current_dst,
                sampler, 
                device=device,
                batch_size=10000,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva
            )
            src_nodes = []
            for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                current_dst=input_nodes
        return current_dst       
        #return affected_edges_by_layer, affected_nodes_by_layer


    def sample_neighbor_infer(
        self, proc_id, device, g, model, use_uva, split_indices,snapshots, batch_size=2048, ksampler=None):
        # updated_edge_size=1024*10
        # snapshots=10
        # split_indices = model.module.updates_initialization(g, device, use_uva, updated_edge_size, snapshots)

        for batch in split_indices:
            final_nodes= model.module.affect_layer_detection(g, device, use_uva, batch, snapshots)
            # Print the device of final_nodes
            logging.debug(f"Device of final_nodes: {final_nodes[0].device if isinstance(final_nodes, list) else final_nodes.device}")
            # Convert final_nodes to the correct device if it's a list
            if isinstance(final_nodes, list):
                final_nodes = final_nodes[0]
            # Ensure       
            dataloader = DataLoader(
                g,
                final_nodes,
                ksampler,
                device=device,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            model.eval()
            ys = []
            y_hats = []
            for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                #logging.debug(f"Number of blocks: {len(blocks)}")
                with torch.no_grad():
                    x = blocks[0].srcdata["feat"]
                    ys.append(blocks[-1].dstdata["label"])
                    y_hats.append(model(blocks, x))
    def cache_neighbor_infer(
        self, proc_id, device, g, model, use_uva, split_indices,snapshots, batch_size=8192, ksampler=None):
        cache_vertex_list=[]
        for l in range(1,len(self.layers)):
            cache_vertices = torch.ones(g.num_nodes(), dtype=torch.bool, device=device)
            cache_vertex_list.append(cache_vertices)
        for batch in split_indices:
            final_nodes= model.module.affect_layer_detection(g, device, use_uva, batch, snapshots)
            fullsampler = MultiLayerFullNeighborSampler(1) 
            curr_nodes=final_nodes
            cached_node_lists=[]  
            uncached_node_lists=[]
            uncached_node_lists.append(curr_nodes)
            for l in range(1,len(self.layers)):
                dataloader = DataLoader(
                    g,
                    curr_nodes,
                    fullsampler,
                    device=device,
                    batch_size=10000,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0,
                    use_ddp=True,
                    use_uva=use_uva,
                )
                for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                    curr_nodes = input_nodes
                    uncached_node_lists.append(input_nodes)

            g.ndata["h"] = g.ndata["feat"]
            fullsampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"]) 
            for l, layer in enumerate(self.layers):
                logging.debug(f"Device of final_nodes: {len(self.layers)-l-1}")
                dataloader = DataLoader(
                    g,
                    uncached_node_lists[len(self.layers)-l-1],
                    fullsampler,
                    device=device,
                    batch_size=10000,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    use_ddp=True,
                    use_uva=use_uva,
                )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
                y = shared_tensor(
                    (
                        g.num_nodes(),
                        self.hid_size
                        if l != len(self.layers) - 1
                        else self.out_size,
                    )
                )
                for input_nodes, output_nodes, blocks in (
                    tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
                ):
                    x = blocks[0].srcdata["h"]
    #                 if l != len(self.layers) - 1:
    # #                    h = F.relu(h)
    #                     kk = self.dropout(x)
    #                 h = layer(blocks[0], kk)  # len(blocks) = 1
                    h = layer(blocks[0], x)  # len(blocks) = 1
                    if l != len(self.layers) - 1:
                        h = self.dropout(h)
                    # non_blocking (with pinned memory) to accelerate data transfer
                    y[output_nodes] = h.to(y.device, non_blocking=True)
                # make sure all GPUs are done writing to 'y'
                dist.barrier()
                g.ndata["h"] = y if use_uva else y.to(device)


    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.hid_size
                    if l != len(self.layers) - 1
                    else self.out_size,
                )
            )
            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                x = blocks[0].srcdata["h"]
#                 if l != len(self.layers) - 1:
# #                    h = F.relu(h)
#                     kk = self.dropout(x)
#                 h = layer(blocks[0], kk)  # len(blocks) = 1
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y



def layerwise_infer(
    proc_id, device, g, num_classes, nid, model, use_uva, batch_size=2**10
):
    
    
    # updated_edge_size=1024
    # snapshots=10
    # batch_size_=8192
    # split_indices = model.module.updates_initialization(g, device, use_uva, updated_edge_size, snapshots)
    
    # samp_sampler = NeighborSampler(
    #     [5, 5], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    # )
    # fullsampler = MultiLayerFullNeighborSampler(
    #         2, prefetch_node_feats=["feat"], prefetch_labels=["label"]
    # )

    # t_test_0 = time.time() 
    # model.module.sample_neighbor_infer(proc_id, device, g, model, use_uva, split_indices,snapshots, batch_size_, samp_sampler)
    # t_test_1 = time.time()
    
    # logging.debug("samp comp  Time {:.5f}".format(t_test_1-t_test_0))  
    
    # t_test_0 = time.time() 
    # model.module.sample_neighbor_infer(proc_id, device, g, model, use_uva,split_indices,snapshots, batch_size_, fullsampler)
    # t_test_1 = time.time()
    # logging.debug("full comp Time {:.5f}".format(t_test_1-t_test_0))   

    # t_test_0 = time.time() 
    # model.module.cache_neighbor_infer(proc_id, device, g, model, use_uva,split_indices,snapshots, batch_size_, fullsampler)
    # t_test_1 = time.time()
    # logging.debug("full Time {:.5f}",t_test_1-t_test_0)   
    # exit(0)
    # acc_x=0
    model.eval()
    with torch.no_grad():
        t_test_0 = time.time() 
        pred = model.module.inference(g, device, batch_size, use_uva)
        t_test_1 = time.time()
        logging.debug("full comp Time {:.5f}".format(t_test_1-t_test_0))   
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)

    dist.barrier()
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=num_classes
        )
        logging.debug("Test accuracy {:.4f}".format(acc.item()))
        acc_x=acc
        
    return acc_x

def evaluate(model, g, num_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def train(
    proc_id,
    nprocs,
    device,
    g,
    num_classes,
    train_idx,
    val_idx,
    test_idx,
    model,
    use_uva,
    num_epochs,
):
    sampler = NeighborSampler(
        [5, 5], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=10000,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=10000,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        for it, (_, _, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"].to(torch.int64)
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
        # opt.zero_grad()
        # total_loss.backward()
        # opt.step()
        acc = (
            evaluate(model, g, num_classes, val_dataloader).to(device) / nprocs
        )
        # acc_test = (
        #     evaluate(model, g, num_classes, test_dataloader).to(device) / nprocs
        # )
        acc_test=0
        
        t1 = time.time()
        dist.reduce(acc, 0)
        if proc_id == 0:
            logging.debug(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Test_Acc {:.4f} | "
                "Time {:.4f}".format(
                    epoch, total_loss / (it + 1), acc.item(), acc_test, t1 - t0
                )
            )


def run(proc_id, nprocs, devices, g, data, mode, num_epochs):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )
    num_classes, train_idx, val_idx, test_idx = data
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    g = g.to(device if mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 128, num_classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing
    use_uva = mode == "mixed"
    # train(
    #     proc_id,
    #     nprocs,
    #     device,
    #     g,
    #     num_classes,
    #     train_idx,
    #     val_idx,
    #     test_idx,
    #     model,
    #     use_uva,
    #     num_epochs,
    # )
    t_test_0 = time.time()
    layerwise_infer(proc_id, device, g, num_classes, test_idx, model, use_uva)
    t_test_1 = time.time()
    logging.debug("layer wise infer time {:.5f}".format(t_test_1-t_test_0))
    # cleanup process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs for train.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ogbn-arxiv",
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/hdd/lhr/dglds",
        help="Root directory of dataset.",
    )
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    logging.debug(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    # load and preprocess dataset
    logging.debug("Loading data")
    dataset = AsNodePredDataset(
        DglNodePropPredDataset(args.dataset_name, root=args.dataset_dir)
    )
    g = dataset[0]
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()
    #if args.dataset_name == "ogbn-arxiv":
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.add_self_loop(g)
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )

    mp.spawn(
        run,
        args=(nprocs, devices, g, data, args.mode, args.num_epochs),
        nprocs=nprocs,
    )

