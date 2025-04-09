import torch

# 假设 edge_index 是一个 (2, num_edges) 的张量
edge_index = torch.tensor([[1, 2, 3, 2, 3, 4],  # 起始节点
                           [0, 0, 0, 1, 2, 1]], dtype=torch.long)  # 目标节点

edge_index.sub

def dst_based_sort(edge_index: torch.Tensor):
    src, dst = edge_index
    dst, indice, counts = dst.unique(
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    indice, reorder = indice.sort()
    src = src[reorder]
    return src, dst, indice, counts


src, dst, indice, counts = dst_based_sort(edge_index)



def compute_batch_range(batch_size: int, nums: torch.Tensor):
    begin = 0
    cum_sum = torch.cat([
        torch.zeros(1, dtype=nums.dtype, device=nums.device),
        torch.cumsum(nums, dim=0)
    ])
    length = len(nums)
    while begin < length:
        target = cum_sum[begin] + batch_size
        end = torch.searchsorted(cum_sum[begin:], target, side='right')
        end = end.item() + begin
        # if end >= length:
        #     break
        # cbegin = cum_sum[begin]
        # cend = cum_sum[end]
        yield begin, end
        begin = end


for begin, end in compute_batch_range(2, counts):
    print(f'from {begin} to {end}')
    print(f'dst: {dst[begin:end]}')
    # print(f'src: {src[cbegin:cend]}')
    # print(f'indice: {indice[cbegin:cend]}')



