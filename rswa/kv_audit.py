"""Optional read-only live block accounting inside the vLLM engine process."""
import json,os
from pathlib import Path

def install():
    dest=os.environ.get('MOSS_KV_AUDIT_PATH')
    if not dest:return
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager,RSWAManager
    if getattr(SingleTypeKVCacheManager,'_moss_audit_installed',False):return
    SingleTypeKVCacheManager._moss_audit_installed=True
    original_allocate=SingleTypeKVCacheManager.allocate_new_blocks
    original_remove=SingleTypeKVCacheManager.remove_skipped_blocks
    original_rswa_remove=RSWAManager.remove_skipped_blocks
    original_free=SingleTypeKVCacheManager.free
    path=Path(dest);path.parent.mkdir(parents=True,exist_ok=True)
    prefixes={};last={}
    def record(manager,request,event,n=None):
        blocks=manager.req_to_blocks.get(request,[])
        key=(id(manager),request)
        if n is None:n=last.get(key,0)
        else:last[key]=n
        p=prefixes.get(key)
        out=dict(event=event,request_id=request,group=manager.kv_cache_group_id,
                 logical_tokens=n,prefix_tokens=p,block_size=manager.block_size,
                 window=getattr(manager,'rswa_window',None),
                 physical_blocks=sum(not b.is_null for b in blocks),logical_blocks=len(blocks),
                 bytes_per_layer_block=manager.kv_cache_spec.page_size_bytes)
        with path.open('a') as f:f.write(json.dumps(out)+'\n')
    def remove_base(self,request_id,processed_computed_tokens,num_prompt_tokens=None):
        prefixes[(id(self),request_id)]=num_prompt_tokens
        return original_remove(self,request_id,processed_computed_tokens,num_prompt_tokens)
    def remove_rswa(self,request_id,processed_computed_tokens,num_prompt_tokens=None):
        prefixes[(id(self),request_id)]=num_prompt_tokens
        return original_rswa_remove(self,request_id,processed_computed_tokens,num_prompt_tokens)
    def allocate(self,request_id,num_tokens,num_tokens_main_model):
        result=original_allocate(self,request_id,num_tokens,num_tokens_main_model)
        p=prefixes.get((id(self),request_id))
        generated=num_tokens-(p or 0)
        last[(id(self),request_id)]=num_tokens
        if generated in (0,1,127,128,129,255,256,257) or generated%256==0:
            record(self,request_id,'allocated',num_tokens)
        return result
    def free(self,request_id):
        record(self,request_id,'request_finished')
        result=original_free(self,request_id)
        prefixes.pop((id(self),request_id),None);last.pop((id(self),request_id),None)
        return result
    SingleTypeKVCacheManager.remove_skipped_blocks=remove_base
    RSWAManager.remove_skipped_blocks=remove_rswa
    SingleTypeKVCacheManager.allocate_new_blocks=allocate
    SingleTypeKVCacheManager.free=free
