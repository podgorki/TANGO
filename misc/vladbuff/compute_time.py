import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import argparse
# import wandb
import random
import numpy as np
from collections import OrderedDict
# from vpr_model import VPRModel
# from utils.validation import get_validation_recalls
# Dataloader
# from dataloaders.val.NordlandDataset import NordlandDataset
# from dataloaders.val.MapillaryDataset import MSLS
# from dataloaders.val.PittsburghDataset import PittsburghDataset
# from dataloaders.val.SPEDDataset import SPEDDataset
from torch.utils.benchmark import Timer
from aggregation import NetVLAD
from salad import SALAD

VAL_DATASETS = [
    "MSLS",
    "pitts30k_test",
    "pitts250k_test",
    "Nordland",
    "SPED",
    "pitts30k_val",
]

def replace_key(k, num):
    for old, new in {'WPCA_'+str(num)+'.':'WPCA.'}.items():
        if old in k:
            k = k.replace(old, new)
    return k

def load_model():
    if "netvlad" in args.aggregation.lower():
        agg_config = args
        useToken = args.useToken_nv
    elif "salad" in args.aggregation.lower():
        agg_config = {
            "num_channels": args.num_channels,
            "num_clusters": args.num_clusters,
            "cluster_dim": args.cluster_dim,
            "token_dim": args.token_dim,
            "expName": args.expName,
            "reduce_feature_dims": args.reduce_feature_dims,
            "reduce_token_dims": args.reduce_token_dims,
            "no_dustbin": args.no_dustbin,
            "args":args,            
        }
        useToken = True
    elif "mixvpr" in args.aggregation.lower():
        agg_config={'in_channels' : args.in_channels,
            'in_h' : args.in_h,
            'in_w' : args.in_w,
            'out_channels' : args.out_channels,
            'mix_depth' : args.mix_depth,
            'mlp_ratio' : args.mlp_ratio,
            'out_rows' : args.out_rows}
        useToken = False

    if "eigenplaces" in args.aggregation.lower():
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                               backbone=args.backbone, fc_output_dim=args.out_dim)        

    elif "cosplace" in args.aggregation.lower():
        model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                               backbone=args.backbone, fc_output_dim=args.out_dim)

    elif "anyloc" in args.aggregation.lower():
        model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", device=args.device)


    elif args.aggregation in ["mixvpr"]:
        model = VPRModel(
            backbone_arch=args.backbone,
            backbone_config={
                'layers_to_crop': [args.layers_to_crop],
            },
            agg_arch=args.aggregation,
            agg_config=agg_config,
            args=args,)
    elif args.aggregation in ["NETVLAD", "SALAD"]:
        model = VPRModel(
            backbone_arch=args.backbone,
            backbone_config={
                "num_trainable_blocks": args.num_trainable_blocks,
                "return_token": useToken,
                "norm_layer": args.norm_layer,
            },
            agg_arch=args.aggregation,
            agg_config=agg_config,
            args=args,
        )

        if args.ckpt_state_dict:
            checkpoint = torch.load(args.resume_train)
            if args.wpca:
                checkpoint["state_dict"] = OrderedDict({replace_key(k, args.num_pcs): v for k, v in checkpoint["state_dict"].items()})
                model_state_dict = model.state_dict()
                # Filter out keys that are not in the model's current state dict
                checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict}

            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(torch.load(args.resume_train))
        model = model.eval()
        model = model.to(args.device)
        print(f"Loaded model from {args.resume_train} Successfully!")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR (SALAD pipeline) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name", type=str, default="gsv_cities", help="Dataset"
    )
    parser.add_argument(
        "--expName", default="0", help="Unique string for an experiment"
    )
    parser.add_argument(
        "--batch_size", type=int, default=60, help="Batch size for the data module"
    )
    parser.add_argument("--img_per_place", type=int, default=4, help="Images per place")
    parser.add_argument(
        "--min_img_per_place", type=int, default=4, help="min_img_per_place"
    )
    parser.add_argument(
        "--shuffle_all",
        type=bool,
        default=False,
        help="Shuffle all images or keep shuffling in-city only",
    )
    parser.add_argument(
        "--random_sample_from_each_place",
        type=bool,
        default=True,
        help="Random sample from each place",
    )
    #    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size (width, height)")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[322, 322],
        help="Resizing shape for images (HxW).",
    )

    parser.add_argument("--num_workers", type=int, default=20, help="Number of workers")
    parser.add_argument(
        "--show_data_stats", type=bool, default=True, help="Show data statistics"
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "resnet", "dinov2_vitg14", "resnet50", "ResNet50", "DINOv2"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--num_trainable_blocks", type=int, default=4, help="Trainable blocks"
    )
    parser.add_argument("--norm_layer", type=bool, default=True, help="Use norm layer")

    # Arguments for helper.py (aggregation configuration)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="SALAD",
        choices=["SALAD", "cosplace", "gem", "convap", "mixvpr", "NETVLAD", "eigenplaces", "anyloc"],
    )
    # Cosplace
    parser.add_argument("--in_dim", type=int, default=2048, help="In dim for cosplace")
    parser.add_argument("--out_dim", type=int, default=512, help="In dim for cosplace/eigenplace")
    # gem
    parser.add_argument("--p", type=int, default=3, help="power for gem")
    # convap
#    parser.add_argument(
 #       "--in_channels", type=int, default=2048, help="in_channels for convap"
    #)
    # MixVPR
    parser.add_argument("--in_channels", type=int, default=1024, help="_")
    parser.add_argument("--in_h", type=int, default=20, help="_")
    parser.add_argument("--in_w", type=int, default=20, help="_")
    parser.add_argument("--out_channels", type=int, default=1024, help="_")
    parser.add_argument("--mix_depth", type=int, default=4, help="_")
    parser.add_argument("--mlp_ratio", type=int, default=1, help="_")
    parser.add_argument("--out_rows", type=int, default=4, help="_")
    parser.add_argument("--layers_to_crop", type=int, default=4, help="_")

    # salad
    parser.add_argument(
        "--num_channels", type=int, default=768, help="num channels for salad"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=64, help="num clusters for salad"
    )
    parser.add_argument(
        "--cluster_dim", type=int, default=128, help="cluster_dim for salad"
    )
    parser.add_argument(
        "--token_dim", type=int, default=256, help="token_dim for salad"
    )
    parser.add_argument(
        "--reduce_feature_dims",
        action="store_true",
        help="Perform dimensionlity reduction for feature",
    )
    parser.add_argument(
        "--no_dustbin",
        action="store_true",
        help="Don't use dustbin",
    )
    parser.add_argument(
        "--reduce_token_dims",
        action="store_true",
        help="Perform dimensionlity reduction for token",
    )
    parser.add_argument(
        "--use_score_cluster",
        action="store_true",
        help="Use cluster layer for SALAD score, cluster centers would be initialized, if False SALAD MLP used",
    )
    parser.add_argument(
        "--no_salad_l2_antiburst_clusters",
        action="store_true",
        help="Dont perform L2 norm in SALAD++ using cluster scoring ",
    )
    parser.add_argument(
        "--salad_antiburst",
        action="store_true",
        help="Perfrom feature-to-feature antiburst within salad",
    )
    # netvlad
    parser.add_argument("--nv_dustbin", action="store_true", help="")    
    parser.add_argument(
        "--l2",
        type=str,
        default="none",
        choices=["before_pool", "after_pool", "onlyFlatten", "none"],
        help="When (and if) to apply the l2 norm with shallow aggregation layers",
    )
    parser.add_argument(
        "--useToken_nv",
        action="store_true",
        help="Use token for NV aggregation",
    )
    parser.add_argument(
        "--fc_output_dim",
        type=int,
        default=512,
        help="Output dimension of final fully connected layer",
    )
    parser.add_argument("--dim", type=int, default=768, help="dim for netvlad")
    parser.add_argument(
        "--clusters_num", type=int, default=64, help="clusters_num for netvlad"
    )
    parser.add_argument(
        "--initialize_clusters",
        action="store_true",
        help="Initialize the cluster for VLAD layer",
    )
    parser.add_argument(
        "--useFC",
        action="store_true",
        help="Add fully connected layer after VLAD layer",
    )
    parser.add_argument(
        "--nv_pca", type=int, help="Use PCA before clustering and nv aggregation."
    )
    parser.add_argument(
        "--nv_pca_randinit",
        action="store_true",
        help="Initialize randomly instead of pca",
    )
    parser.add_argument(
        "--nv_pca_alt", action="store_true", help="use fc layer instead of pca"
    )
    parser.add_argument(
        "--nv_pca_alt_mlp", action="store_true", help="use 2-fc layer mlp layer instead of pca / pca_alt"
    )

    # ab params
    parser.add_argument("--ab_upsample", action="store_true", help="")
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=16,
        help="Batch size for inference (validating and testing)",
    )
    parser.add_argument(
        "--antiburst",
        action="store_true",
        help="use self sim + sigmoid to remove burstiness",
    )
    parser.add_argument("--ab_w", type=float, default=8.0, help="")
    parser.add_argument("--ab_b", type=float, default=7.0, help="")
    parser.add_argument("--ab_p", type=float, default=1.0, help="")
    parser.add_argument(
        "--ab_gen", type=int, help="generates thresholds from soft_assign"
    )
    parser.add_argument("--ab_relu", action="store_true", help="")
    parser.add_argument(
        "--ab_soft",
        action="store_true",
        help="softmax instead of sigmoid before summing",
    )
    parser.add_argument("--ab_inv", action="store_true", help="")
    parser.add_argument("--ab_t", type=float, help="thresh for relu")
    parser.add_argument("--ab_testOnly", action="store_true", help="")
    parser.add_argument("--ab_allFreezeButAb", action="store_true", help="")
    parser.add_argument(
        "--ab_fixed",
        action="store_true",
        help="ab params are init but arent nn.Parameter",
    )
    parser.add_argument(
        "--ab_kp", type=int, help="num middle dim for fc-relu-fc weight per pixel"
    )
    parser.add_argument(
        "--ab_wOnly", action="store_true", help="train w, freeze b and p as init"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_"
    )
    #    parser.add_argument("--ckpt_path", type=str, required=True, default=None, help="Path to the checkpoint")
    parser.add_argument(
        "--resume_train",
        type=str,
        required=False,
        default=None,
        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth",
    )
    parser.add_argument(
        "--ckpt_state_dict", action="store_true", help="Use checkpoint state dictionary"
    )
    # Datasets parameters
    parser.add_argument(
        "--val_datasets",
        nargs="+",
        default=VAL_DATASETS,
        help="Validation datasets to use",
        choices=VAL_DATASETS,
    )
    parser.add_argument(
        "--wpca",
        action="store_true",
        help="Use post pool WPCA layer",
    )
    parser.add_argument(
        "--pl_seed",
        action="store_true",
        help="Use pytorch_lightning (pl) seed, default is pytorch seeding",
    )
    parser.add_argument(
        "--get_flops",
        action="store_true",
        help="Use to capure the flops for backbone and aggregators",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_pcs", type=int, default=512, help="Use post pool PCA.")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Dont use wandb",
    ) 
    args = parser.parse_args()

    # Parse image size
    if args.resize:
        if len(args.resize) == 1:
            args.resize = (args.resize[0], args.resize[0])
        elif len(args.resize) == 2:
            args.resize = tuple(args.resize)
        else:
            raise ValueError("Invalid image size, must be int, tuple or None")

        args.resize = tuple(map(int, args.resize))

    return args


if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = True
    args = parse_args()
    if args.aggregation.lower()=='netvlad':
        print(f'Clusters_num / dim: {args.clusters_num}/{args.dim}') 
    if args.aggregation.lower()=='salad':
        print(f'Token / Feat Dim Red: {args.reduce_token_dims}/{args.reduce_feature_dims}')
        print(f'Num_clusters / Cluster_dim:{args.num_clusters}/{args.cluster_dim}')
    print(f'Aggregation: {args.aggregation}, AntiBurst', args.antiburst)
    print(f'NV_PCA: {args.nv_pca}, NV_PCA_ALT: {args.nv_pca_alt}, NV_PCA_ALT_MLP: {args.nv_pca_alt_mlp}')


    dataset_name = args.dataset_name.lower()[:4]
    wandb_dataStr = dataset_name
    args.expName = "eval-" + wandb_dataStr + args.expName

    # Check: dont have wpca for systems having fc layer after NV layer
    if args.useFC:
        assert args.wpca == False

    if args.pl_seed:
        pl.utilities.seed.seed_everything(seed=int(args.seed), workers=True)
    else:        
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if args.device == 'cuda':
            #noinspection PyUnresolvedReferences
            torch.cuda.manual_seed(int(args.seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

    # wandb.init(project="vgBench-CosPlace-SALAD", config=args, entity="vpr-sg")
    # # update runName
    # runName = wandb.run.name
    # wandb.run.name = args.expName + "-" + runName.split("-")[-1]
    # wandb.run.save()


    from torch.utils.benchmark import Timer
    import torch
    from tqdm import tqdm

    def measure_aggregator(model, input_tensor):
        return model(input_tensor)

    # model = load_model()
    if args.aggregation.lower() == "netvlad":
        model = NetVLAD(clusters_num=args.clusters_num,dim=args.dim,args=args).cuda()
    elif args.aggregation.lower() == "salad":
        model = SALAD(num_channels=args.num_channels, num_clusters=args.num_clusters, cluster_dim=args.cluster_dim, token_dim=args.token_dim).cuda()

    num_runs = 100
    total_time = 0  # To accumulate total time taken

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float):
            for _ in tqdm(range(num_runs), "Calculating descriptors..."):
                # Generate a random tensor for each iteration
                image_encoding = torch.randn((64, 768, 23, 23), device=args.device)
                if args.aggregation.lower() == "salad":
                    image_encoding_2 = torch.randn((64, 768), device=args.device)
                    image_encoding = [image_encoding, image_encoding_2]
                
                # Define the timer and measure the execution time
                timer = Timer(
                    stmt="measure_aggregator(model, image_encoding)",
                    globals={"measure_aggregator": measure_aggregator, "model": model, "image_encoding": image_encoding}
                )

                # Measure and accumulate the time taken for one run
                total_time += timer.timeit(number=1).mean  # .mean gives the average time per run in seconds

    # Calculate the average time per iteration in microseconds
    average_time = (total_time / num_runs) * 1e3  # Convert seconds to microseconds
    # wandb.log({'average_time': average_time:.2f})
    print(f"Average time per iteration: {average_time:.2f} ms")


