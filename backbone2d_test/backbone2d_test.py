import torch
from torch import nn
from backbone2d_model import LDFCNet_waymo,LDFCNet_nuscenes, LDFCNet_puls_nuscense, BaseBEVResBackbone, BaseBEVBackbone, CascadeDEDBackbone

@torch.no_grad()
def _profile(model, batch_dict):
    from thop import profile
    flops, params = profile(model, (batch_dict,))
    print('totle Gflops: %.2f M, totle params: %.2f M' % (flops / 10 ** 9, params / 1e6))

@torch.no_grad()
def _compute_runtime(network, batch_dict, warm=1000, loop=5000):
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for i in range(warm):
        pred_dicts, ret_dict = network(batch_dict)

    total_time = 0
    for i in range(loop):
        start_event.record()
        pred_dicts, ret_dict = network(batch_dict)
        end_event.record()
        start_event.synchronize()
        end_event.synchronize()
        time = start_event.elapsed_time(end_event)
        total_time += time
        print(f"{time:.2f}", end=" ")
        if (i+1) % 20 == 0:
            print()
    avg_time = total_time / loop
    print(f"avg: {avg_time}")


if __name__ == '__main__':
    device = 'cuda:0'
    # device = 'cpu'

    model = BaseBEVBackbone(None, 128).to(device)
    # model = BaseBEVResBackbone(None, 128).to(device)
    # model = CascadeDEDBackbone(None, 128).to(device)

    def replace_batchnorm(net):
        for child_name, child in net.named_children():
            if hasattr(child, 'fuse'):
                fused = child.fuse()
                setattr(net, child_name, fused)
                replace_batchnorm(fused)
            else:
                replace_batchnorm(child)
    
    replace_batchnorm(model)

    data_dict = {}
    data_dict['spatial_features'] = torch.randn([1, 128, 236, 236], dtype=torch.float32, device=device)

    # _profile(model, data_dict)
    _compute_runtime(model, data_dict)

