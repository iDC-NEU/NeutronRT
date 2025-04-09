import torch
import time

from lib.cpp_extension.wrapper import gpuDataCache


class GPUCache:

    def __init__(self, num_nodes_, host_key_, host_emb_, 
              dev_key_, dev_emb_, device, in_GPU=True):
        host_key_ = host_key_.to(device)
        dev_key_ = dev_key_.to(device)

        # hold the reference of embeddings
        self.host_emb = host_emb_.pin_memory()
        self.dev_emb = dev_emb_.to(device)
        self.device = device
        self.cache = gpuDataCache.GPUDataCache(num_nodes_, 
                        host_key_, self.host_emb,
                        dev_key_, self.dev_emb, device.index, in_GPU)
        self.hotness = torch.zeros(num_nodes_, dtype=torch.int64, device=device)
        self.cache_type = torch.zeros(num_nodes_, dtype=torch.int8, device=device)
        self.cache_type[host_key_] = 1
        self.cache_type[dev_key_] = 2

        self.empty_float32 = torch.empty(0, dtype=torch.float32, device=device)
        self.empty_bool = torch.empty(0, dtype=torch.bool, device=device)
    

    def query(self, ids_, counts_ids_=1, out=None, ignore=True):
        if isinstance(counts_ids_, int):
            if counts_ids_ == 0:
                counts_ids_ = torch.zeros_like(ids_, device=self.device)
            elif counts_ids_ == 1:
                counts_ids_ = torch.ones_like(ids_, device=self.device)
        if out is None:
            out = torch.zeros((ids_.size(0), self.num_feat),
                        dtype=torch.float32, device=self.device)
        miss = torch.ones_like(ids_, dtype=torch.bool, device=self.device)
        ret = self.cache.query(ids_, counts_ids_, out, miss, ignore)
        if not ret:
            raise RuntimeError("Error: query failed")
        return out, miss
    
    @property
    def num_feat(self):
        return self.dev_emb.size(1)
    
    @property
    def num_dev_items(self):
        return self.dev_emb.size(0)
    
    @property
    def num_host_items(self):
        return self.host_emb.size(0)
    
    def check_state(self, keys, ignore=True) -> torch.Tensor:
        return self.cache.check_state(keys, ignore)

    def invalidate(self, keys):
        return self.cache.invalidate(keys)

    def replace(self, ignore_host, l_keys=[], l_values=[]):
        self.cache.update_hotness(self.hotness) # hotness * keys masks (0/1) (data in cache / host + data in keys / keys_2)
        if ignore_host:
            final_hotness = self.hotness
        else:
            keys_mask = torch.zeros_like(self.hotness)
            keys_mask[self.cache_type > 0] = 1
            for keys in l_keys:
                keys_mask[keys] = 1
            tmp_hotness = self.hotness + keys_mask
            final_hotness = tmp_hotness * keys_mask

        _, node_ids = torch.sort(final_hotness, descending=True)
        if ignore_host:
            new_cache = torch.ones_like(self.cache_type)
        else:
            new_cache = torch.zeros_like(self.cache_type)
        new_cache[node_ids[:self.num_dev_items]] = 2
        if not ignore_host:            
            new_cache[node_ids[self.num_dev_items:self.num_dev_items+self.num_host_items]] = 1
            # for verify
            # if not torch.equal(final_hotness, tmp_hotness):
            #     tmp = tmp_hotness[final_hotness != tmp_hotness].max()
            #     tmp2 = tmp_hotness[node_ids[self.num_dev_items+self.num_host_items-1]]
            #     if tmp > tmp2:
            #         print(f"Warning: not found used keys {tmp} {tmp2}")
        dev_new = torch.nonzero(torch.logical_and(new_cache == 2, self.cache_type != 2)).to(torch.int64).squeeze()
        if ignore_host and dev_new.nelement() == 0:
            return True
        
        dev_replace = torch.nonzero(torch.logical_and(new_cache != 2, self.cache_type == 2)).to(torch.int64).squeeze()
        if ignore_host:
            ret = self.cache.replace_from_host(dev_replace, dev_new)
        else:
            dev_new_pos, nq_masks = self.get_from_kvs(l_keys, dev_new)
            if nq_masks.any():
                host_values, miss = self.query(dev_new[nq_masks], 0)
                if miss.any():
                    raise RuntimeError("Error: replace failed")
                dev_new_pos[nq_masks] = torch.arange(host_values.size(0), device=self.device)
            else:
                host_values = self.empty_float32     

            host_new = torch.nonzero(torch.logical_and(new_cache == 1, self.cache_type != 1)).to(torch.int64).squeeze()
            host_new_pos, nq_masks = self.get_from_kvs(l_keys, host_new)
            # for test
            # if nq_masks.any():
            #     miss = self.check_state(host_new[nq_masks]) < 0
            #     if miss.any():
            #         raise RuntimeError("Error: replace failed")

            host_replace = torch.nonzero(torch.logical_and(new_cache != 1, self.cache_type == 1)).to(torch.int64).squeeze()
            ret = self.cache.replace(l_keys, l_values, 
                        host_replace, host_new, host_new_pos, 
                        dev_replace, dev_new, dev_new_pos, host_values)

        self.cache_type = new_cache
        return ret


    def get_from_kvs(self, l_keys, dev_new):
        dev_new_pos = torch.empty_like(dev_new)
        not_equal_mask = torch.ones_like(dev_new, dtype=torch.bool)
        for step, keys in enumerate(l_keys):
            if step == 0:
                torch.searchsorted(keys, dev_new, out=dev_new_pos)
                not_equal_mask = self.update_masks(keys, dev_new, dev_new_pos, not_equal_mask)
            else:
                remain = dev_new[not_equal_mask]
                dev_new_left = torch.searchsorted(keys, remain)
                dev_new_pos[not_equal_mask] = dev_new_left
                not_equal_mask = self.update_masks(keys, remain, dev_new_left, not_equal_mask)
            if not not_equal_mask.any():
                break
        return dev_new_pos, not_equal_mask


    def update_masks(self, keys, dev_new, dev_new_pos, not_equal):
        if dev_new_pos.nelement() == 0:
            return not_equal
        elif dev_new_pos.nelement() == 1:
            last_one = int(dev_new_pos)
        else:
            last_one = dev_new_pos[-1] 
        masks = not_equal.clone()
        if last_one < keys.nelement():
            tmp_equal = dev_new != keys[dev_new_pos]
        else:
            tmp_equal = dev_new_pos >= keys.nelement()
            compare_pos = torch.where(tmp_equal)[0][0]
            if compare_pos > 0:
                tmp_equal[:compare_pos] = dev_new[:compare_pos] != keys[dev_new_pos[:compare_pos]]
        masks[not_equal] = tmp_equal
        return masks

    def sync_replace(self):
        return self.cache.sync_replace()
    
    def update(self, keys, values, ignore=None):
        if ignore is None:
            ignore = self.empty_bool
        return self.cache.update(keys, values, ignore)
    
    def sync_update(self):
        return self.cache.sync_update()
    

def check_part_compute_state(aggr_cache, emb_cache, aggr_dst, emb_src,
        src_indices, o_type):
    can_part = torch.zeros_like(aggr_dst, dtype=torch.bool)
    need_src = torch.zeros_like(emb_src, dtype=torch.bool)
    if not gpuDataCache.check_part_compute_state(aggr_cache.cache, emb_cache.cache, 
                                        aggr_dst, emb_src, src_indices, o_type, 
                                        can_part, need_src):
        raise RuntimeError("Error: check_part_compute_state failed")
    return can_part, need_src
