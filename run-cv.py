import argparse

import pandas as pd
import torch
import os
import numpy as np
from torch import nn
from torch.optim import Adagrad
from torch.utils.tensorboard import SummaryWriter

from utils import get_training_state_catgwr, LoopPhase, BINARIES_PATH, CHECKPOINTS_PATH, \
    get_available_binary_name, RESULTS_PATH, get_result_name
from data_process import DataProcessor
from models import CatGWRegression
from tqdm import trange

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_TEST_RATE = 0.15


def calc_spatial_weight(edge_distances):
    """
    Calculate spatial weights for each edge with a gussian kernel
    :param edge_distances: Tensor, edge distances matrix
    :return: spatial weights
    """

    max_distance = torch.max(edge_distances, dim=1).values
    spatial_weights = torch.exp(-1.0 * (edge_distances / max_distance) ** 2)
    spatial_weights[spatial_weights == 1] = .0

    return spatial_weights


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=3000)
    parser.add_argument("--early_stop_begin", type=int, help="number of epochs when early stop is begin", default=1000)
    parser.add_argument("--patience_period", type=int,
                        help="number of epochs with no improvement on val before terminating", default=500)
    parser.add_argument("--lr", type=float, help="model learning rate", default=0.1)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=2e-4)
    parser.add_argument("--num_of_heads", type=int, help="number of attention heads", default=2)
    parser.add_argument("--dropout_prob", type=float, help="dropout probability", default=0.3)
    parser.add_argument("--should_test", type=bool, help='should test the model on the test dataset?', default=True)
    parser.add_argument("--cv_folds", type=int, help="number of cross validation folds", default=10)

    # Dataset related
    parser.add_argument("--dataset_name", type=str, help='dataset name', default='HousePrice')
    parser.add_argument("--rand_seed", type=int, help='random seed for data shuffling, train/val/test set split',
                        default=11)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=True)
    parser.add_argument("--console_log_freq", type=int,
                        help="log to output console (epoch) freq (None for no logging)", default=500)
    parser.add_argument("--checkpoint_freq", type=int,
                        help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
    parser.add_argument("--output_sw", type=bool, help="enable output spatial weight to result excel", default=False)

    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    print('====================================================')
    print('             Training configuration                 ')
    print('====================================================')
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
        print(f"{arg:<25} {training_config[arg]}")
    print('====================================================\n')

    return training_config


def get_main_loop(config, model, loss_func, optimizer, edge, vertex_feature, vertex_y, vertex_x, train_indices,
                  val_indices, test_indices, patience_period):
    node_dim = 0

    train_ys = vertex_y.index_select(node_dim, train_indices)
    val_ys = vertex_y.index_select(node_dim, val_indices)
    test_ys = vertex_y.index_select(node_dim, test_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (vertex_feature, edge)  # Pack data into tuples

    def get_node_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices
        elif phase == LoopPhase.VAL:
            return val_indices
        elif phase == LoopPhase.TEST:
            return test_indices
        else:
            torch.cat((train_indices, val_indices, test_indices), 0)

    def get_vertex_ys(phase):
        if phase == LoopPhase.TRAIN:
            return train_ys
        elif phase == LoopPhase.VAL:
            return val_ys
        elif phase == LoopPhase.TEST:
            return test_ys
        else:
            return vertex_y

    def final_run():
        model.eval()
        betas, y_hat, spatial_weight = model(graph_data)
        spatial_weight = spatial_weight.squeeze(0)

        indicators = torch.Tensor(1, 4).cuda()
        for phase in LoopPhase:
            node_indices = get_node_indices(phase)
            gt_vertex_ys = get_vertex_ys(phase)
            y_hat_phase = y_hat.index_select(node_dim, node_indices).squeeze()
            # statistical indicators:
            rmse = torch.sqrt(nn.MSELoss()(y_hat_phase, gt_vertex_ys))
            mae = nn.L1Loss()(y_hat_phase, gt_vertex_ys)
            mape = torch.mean(torch.abs((y_hat_phase - gt_vertex_ys) / gt_vertex_ys))
            rss = torch.sum((y_hat_phase - gt_vertex_ys) ** 2)
            rsquare = 1 - rss / torch.sum((gt_vertex_ys - torch.mean(gt_vertex_ys)) ** 2)
            adj_rsquare = 1 - (1 - rsquare) * (len(gt_vertex_ys) - 1) / (len(gt_vertex_ys) - vertex_x.shape[1] - 1)
            indicator = torch.cat((rss.reshape(1), rmse.reshape(1), mae.reshape(1), mape.reshape(1), rsquare.reshape(1),
                                   adj_rsquare.reshape(1)), 0)
            indicators = torch.vstack((indicators, indicator)) if indicators.shape[0] > 1 else indicator

        return y_hat, betas, indicators, spatial_weight

    def main_loop(phase, epoch=0, writer=None):
        global BEST_RES, BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT

        if phase == LoopPhase.TRAIN:
            model.train()
        else:
            model.eval()

        node_indices = get_node_indices(phase)
        gt_vertex_ys = get_vertex_ys(phase)  # gt stands for ground truth

        betas, y_hat, _ = model(graph_data)
        y_hat = y_hat.index_select(node_dim, node_indices).squeeze()

        loss = torch.sqrt(loss_func(y_hat, gt_vertex_ys))

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        accuracy = 1 - torch.sum((y_hat - gt_vertex_ys) ** 2) / torch.sum(
            (gt_vertex_ys - torch.mean(gt_vertex_ys)) ** 2)

        if phase == LoopPhase.TRAIN:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), epoch)
                writer.add_scalar('training_acc', accuracy, epoch)

            # Save model checkpoint
            if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0:
                ckpt_model_name = f"catgwr_ckpt_epoch_{epoch + 1}.pth"
                config['test_acc'] = -1
                torch.save(get_training_state_catgwr(config, model), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

        elif phase == LoopPhase.VAL:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('val_loss', loss.item(), epoch)
                writer.add_scalar('val_acc', accuracy, epoch)

            # Log to console
            if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                # get the current learning rate:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f'\nNow training: epoch={epoch + 1} | lr= {current_lr} | val loss={(loss.item()):.4f} | val R2={(accuracy):.4f}')

            # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
            # or the val loss keeps going down we won't stop
            if epoch > config['early_stop_begin']:
                if loss.item() < BEST_VAL_LOSS and accuracy > BEST_VAL_ACC:
                    BEST_VAL_ACC = max(accuracy, BEST_VAL_ACC)  # keep track of the best validation accuracy so far
                    BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
                    PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
                else:
                    PATIENCE_CNT += 1  # otherwise keep counting

                if PATIENCE_CNT >= patience_period:
                    raise Exception('\nStopping the training, the universe has no more patience for this training.')

        elif phase == LoopPhase.TEST:
            return accuracy  # in the case of test phase we just report back the test accuracy

    return main_loop, final_run  # return the decorated function


def train_model(config):
    global BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: load the graph data

    edge_path = './data/simu_data_S1-dist400.csv'
    vertex_path = './data/simu_data_S1.csv'
    x_column = ['x1', 'x2']
    y_column = 'y'
    feature_column = ['e1', 'e2', 'e3', 'e4']

    data_processer = DataProcessor()

    edge, vertex_feature, vertex_y, vertex_x = data_processer.load_all_data(edge_path, vertex_path,
                                                                            feature_column, y_column, x_column,
                                                                            is_normalize_feature=False,
                                                                            is_normalize_variable=False,
                                                                            device=device)

    # Step 2: prepare the model
    spatial_weights = calc_spatial_weight(edge)
    ols_weights = torch.matmul(
        torch.matmul(torch.inverse(torch.matmul(vertex_x.transpose(0, 1), vertex_x)), vertex_x.transpose(0, 1)),
        vertex_y)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)

    loss_fn = nn.MSELoss(reduction='mean')

    indicator_df = pd.DataFrame(columns=['dataset', 'RSS', 'RMSE', 'MAE', 'MAPE', 'R2', 'adj-R2'])

    node_num = len(vertex_y)
    test_num = int(DATA_TEST_RATE * node_num)
    test_indices = torch.arange(node_num - test_num, node_num, dtype=torch.long, device=device)
    for icv in range(config['cv_folds']):

        model = CatGWRegression(
            num_of_nodes=len(vertex_y),
            num_of_heads=config['num_of_heads'],
            num_of_features=len(feature_column),
            num_of_variables=len(x_column) + 1,
            spatial_weights=spatial_weights,
            vertex_y=vertex_y,
            vertex_x=vertex_x,
            ols_weights=ols_weights,
            dropout_prob=config['dropout_prob']
        ).to(device)

        optimizer = Adagrad(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        if config['cv_folds'] == 1:
            val_rate = 1 / 10
        else:
            val_rate = 1 / config['cv_folds']

        train_val_num = node_num - test_num

        val_indices = torch.arange(icv * int(train_val_num * val_rate), (icv + 1) * int(train_val_num * val_rate),
                                   dtype=torch.long, device=device)
        train_indices = torch.cat(
            [torch.arange(0, icv * int(train_val_num * val_rate), dtype=torch.long, device=device),
             torch.arange((icv + 1) * int(train_val_num * val_rate), node_num - test_num, dtype=torch.long,
                          device=device)])

        main_loop, final_run = get_main_loop(
            config,
            model,
            loss_fn,
            optimizer,
            edge,
            vertex_feature,
            vertex_y,
            vertex_x,
            train_indices,
            val_indices,
            test_indices,
            config['patience_period'])

        BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT = [-np.inf, np.inf, 0]  # reset vars used for early stopping

        cvstr = f'_cv{icv}'
        writer = SummaryWriter(comment=cvstr)

        # Step 4: Start the training procedure
        for epoch in trange(0, config['num_of_epochs']):
            # for epoch in range(config['num_of_epochs']):
            # Training loop
            main_loop(phase=LoopPhase.TRAIN, epoch=epoch, writer=writer)

            # Validation loop
            with torch.no_grad():
                try:
                    main_loop(phase=LoopPhase.VAL, epoch=epoch, writer=writer)
                except Exception as e:  # "patience has run out" exception :O
                    print(str(e))
                    break  # break out from the training loop

        # Step 5: Test or predict

        if config['should_test']:
            test_acc = main_loop(phase=LoopPhase.TEST)
            config['test_acc'] = test_acc
            print(f'Test accuracy = {test_acc}')
        else:
            config['test_acc'] = -1

        # Step 6: Get the regression results and statistical indicators
        if config['output_sw']:
            y_hat, betas, indicators, spatial_weight = final_run()
        else:
            y_hat, betas, indicators, _ = final_run()
            spatial_weight = None
        result_name = get_result_name(vertex_path.split('/')[-1].split('.')[0], icv)
        path = os.path.join(RESULTS_PATH, result_name)
        indicator_df = pd.concat([indicator_df,
                                  data_processer.print_and_output_result(path, result_name, y_hat, indicators, icv,
                                                                         betas, spatial_weight)], ignore_index=True,
                                 axis=0)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 5000)

        # Step 7: Save the trained model (weights)
        # Save the latest model in the binaries directory
        torch.save(get_training_state_catgwr(config, model), os.path.join(BINARIES_PATH, get_available_binary_name()))
        print()
    print(f'************ CROSS VALIDATION RESULTS ************')
    print(indicator_df)


if __name__ == '__main__':
    train_model(get_training_args())
