"""Conservative cgroup-v1 headroom, separating cold clean file cache."""
from pathlib import Path


def container_budget(root=Path('/sys/fs/cgroup/memory')):
    limit=int((root/'memory.limit_in_bytes').read_text())
    used=int((root/'memory.usage_in_bytes').read_text())
    values={key:int(value) for key,value in (line.split() for line in (root/'memory.stat').read_text().splitlines())}
    def value(name):return values.get('total_'+name,values.get(name,0))
    clean_inactive=max(0,value('inactive_file')-value('dirty')-value('writeback'))
    unit=1024**3
    return {'container_limit_gib':limit/unit,'container_usage_gib':used/unit,
            'container_headroom_gib':(limit-used)/unit,'clean_inactive_file_gib':clean_inactive/unit,
            'conservative_available_gib':(limit-used+.5*clean_inactive)/unit,
            'cache_policy':'Credit only half inactive_file minus dirty/writeback; never all cache or host MemAvailable alone'}
