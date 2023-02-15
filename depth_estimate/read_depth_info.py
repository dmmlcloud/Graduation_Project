import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)
from mmcv.image import tensor2imgs

from depth.datasets import build_dataloader, build_dataset
from depth.models import build_depther

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='depth test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def replace_str(s):
    if s[0] == '/':
        return s[1:]
    new_str = s.replace('/', '_')
    return new_str

def single_gpu_depth(model,
                    data_loader,
                    show=False,
                    out_dir=None):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    # when none of them is set true, return depth results as
    # a list of np.array.

    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    results = []
    for _, data in zip(loader_indices, data_loader):

        with torch.no_grad():
            result_depth = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    mmcv.mkdir_or_exist(out_dir)
                    out_file = osp.join(out_dir, replace_str(img_meta['ori_filename']))
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result_depth,
                    show=show,
                    out_file=out_file)
        results.append(result_depth)
        prog_bar.update()

    return results

def main():
    args = parse_args()

    assert args.out or args.show or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    
    if args.out:
        print(os.path.dirname(args.out))
        mmcv.mkdir_or_exist(os.path.dirname(args.out))
    
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    print("config:" , cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None

    model = build_depther(
        cfg.model,
        test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # for other models
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    model = MMDataParallel(model, device_ids=[0])
    depth_results = single_gpu_depth(
        model,
        data_loader,
        args.show,
        args.show_dir)
    print("\n\ndepth results showing below:")
    for index, depth in enumerate(depth_results):
        print("(", index+1, "): ", depth)

if __name__ == '__main__':
    main()