import argparse


def settings():
    parser = argparse.ArgumentParser()

    # Public parameters
    parser.add_argument('--seed', type=int, default=0, help='Random seed. Default is 0.')
    # no-cuda 被触发则为action，即True，否则为False
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--in_file', required=True, help='Path to data fold. e.g., data/DDI.edgelist')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers. Default is 0.')
    parser.add_argument('--batch', type=int, default=256, help='Batch size. Default is 256.')
    parser.add_argument('--feature_type', type=str, default='position',
                        choices=['one_hot', 'uniform', 'normal', 'position'],
                        help='Initial node feature type. Default is position.')
    parser.add_argument('--aggregator', choices=['GCN', 'GIN'], default='GCN',
                        help='Message passing framework adopted.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate. Default is 5e-4.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1-keep probability). Default is 0.5.')
    parser.add_argument('--weight_decay', default=5e-4, help='Weight decay (L2 loss on parameters). Default is 5e-4.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train. Default is 30.')

    parser.add_argument('--train_ratio', type=float, default=0.7, help='Remain links in network. Default is 1.')
    parser.add_argument('--out_file', required=True, help='Path to data result file. e.g., result.txt')

    parser.add_argument('--loss_ratio1', type=float, default=1, help='Ratio of task 1. Default is 1.')
    parser.add_argument('--loss_ratio2', type=float, default=0.1, help='Ratio of task 2. Default is 0.1.')
    parser.add_argument('--loss_ratio3', type=float, default=0.1, help='Ratio of task 3. Default is 0.1.')

    # GCN parameters
    parser.add_argument('--dimensions', type=int, default=128, help='Dimensions of feature. Default is 128.')
    parser.add_argument('--hidden1', default=64,
                        help='Number of hidden units for encoding layer 1 for CSGNN. Default is 64.')

    parser.add_argument('--hidden2', default=32,
                        help='Number of hidden units for encoding layer 2 for CSGNN. Default is 32.')

    parser.add_argument('--decoder1', default=512,
                        help='Number of hidden units for decoding layer 1 for CSGNN. Default is 512.')


    # AFGRL parameters
    parser.add_argument('--dropout_afgrl', type=float, default=0.5, help='Dropout rate for AFGRL. Default is 0.5.')
    parser.add_argument('--mad', type=float, default=0.9, help='Moving average decay for teacher network.')
    parser.add_argument('--epochs_afgrl', type=int, default=100, help='Number of epochs to train. Default is 100.')
    parser.add_argument('--device', type=int, default=0, help='The id of cuda to be used.')
    parser.add_argument('--num_centroids', type=int, default=100,
                        help='The number of centroids for K-means Clustering.')
    parser.add_argument('--num_kmeans', type=int, default=5,
                        help='The number of k-means clustering for being robust to randomness.')
    parser.add_argument('--clus_num_iters', type=int, default=20, help='Number of training iterations for KMeans.')
    parser.add_argument('--pred_hid', type=int, default=2048,
                        help='Number of hidden units in the predictor. Default is 512')
    parser.add_argument("--topk", type=int, default=4, help="The number of neighbors to search")
    parser.add_argument('--decoder_dim1', default=512,
                        help='Number of hidden units for decoding layer 1 for CSGNN. Default is 512.')

    args = parser.parse_args()
    return args