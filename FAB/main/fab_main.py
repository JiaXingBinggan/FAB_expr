import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import random
import FAB.models.RL_brain_fab as td3

import sklearn.preprocessing as pre

import tqdm

import torch
import torch.nn as nn
import torch.utils.data

from itertools import islice
from FAB.config import config
import logging
import sys

np.seterr(all='raise')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def bidding(bid):
    return int(bid if bid <= 300 else 300)


def generate_bid_price(datas):
    '''
    :param datas: type list
    :return:
    '''
    return np.array(list(map(bidding, datas))).astype(int)


def bid_main(bid_prices, imp_datas, budget):
    '''
    主要竞标程序
    :param bid_prices:
    :param imp_datas:
    :return:
    '''
    win_imp_indexs = np.where(bid_prices >= imp_datas[:, 2])[0]

    win_imp_datas = imp_datas[win_imp_indexs, :]

    win_clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
    if len(win_imp_datas):
        first, last = 0, win_imp_datas.shape[0] - 1

        final_index = 0
        while first <= last:
            mid = first + (last - first) // 2
            tmp_sum = np.sum(win_imp_datas[:mid, 2])
            if tmp_sum < budget:
                first = mid + 1
            else:
                last_sum = np.sum(win_imp_datas[:mid - 1, 2])
                if last_sum <= budget:
                    final_index = mid - 1
                    break
                else:
                    last = mid - 1
        final_index = final_index if final_index else first
        win_clks = np.sum(win_imp_datas[:final_index, 0])
        origin_index = win_imp_indexs[final_index - 1]

        real_clks = np.sum(imp_datas[:origin_index, 0])
        imps = final_index + 1
        bids = origin_index + 1

        cost = np.sum(win_imp_datas[:final_index, 2])
        current_cost = np.sum(win_imp_datas[:final_index, 2])

        if len(win_imp_datas[final_index:, :]) > 0:
            if current_cost < budget:
                budget -= current_cost

                final_imps = win_imp_datas[final_index:, :]
                lt_budget_indexs = np.where(final_imps[:, 2] <= budget)[0]

                final_mprice_lt_budget_imps = final_imps[lt_budget_indexs]
                last_win_index = 0
                for idx, imp in enumerate(final_mprice_lt_budget_imps):
                    tmp_mprice = final_mprice_lt_budget_imps[idx, 2]
                    if budget - tmp_mprice >= 0:
                        win_clks += final_mprice_lt_budget_imps[idx, 0]
                        imps += 1
                        bids += (lt_budget_indexs[idx] - last_win_index + 1)
                        last_win_index = lt_budget_indexs[idx]
                        cost += tmp_mprice
                        budget -= tmp_mprice
                    else:
                        break
                real_clks += np.sum(final_imps[:last_win_index, 0])
            else:
                win_clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
                last_win_index = 0
                for idx, imp in enumerate(win_imp_datas):
                    tmp_mprice = win_imp_datas[idx, 2]
                    real_clks += win_imp_datas[idx, 0]
                    if budget - tmp_mprice >= 0:
                        win_clks += win_imp_datas[idx, 0]
                        imps += 1
                        bids += (win_imp_indexs[idx] - last_win_index + 1)
                        last_win_index = win_imp_indexs[idx]
                        cost += tmp_mprice
                        budget -= tmp_mprice

    return win_clks, real_clks, bids, imps, cost


def get_model(args, device):
    RL_model = td3.TD3_Model(args.neuron_nums,
                             action_nums=1,
                             lr_A=args.lr_A,
                             lr_C=args.lr_C,
                             memory_size=args.memory_size,
                             tau=args.tau,
                             batch_size=args.rl_batch_size,
                             device=device
                             )

    return RL_model


def get_dataset(args):
    data_path = args.data_path + args.dataset_name + args.campaign_id

    # clk,ctr,mprice,hour,time_frac
    columns = ['clk', 'ctr', 'mprice', 'hour', 'time_frac']
    train_data = pd.read_csv(data_path + 'train.bid.' + args.sample_type + '.data')[columns]
    test_data = pd.read_csv(data_path + 'test.bid.' + args.sample_type + '.data')[columns]

    train_data = train_data[['clk', 'ctr', 'mprice', 'hour']].values.astype(float)
    test_data = test_data[['clk', 'ctr', 'mprice', 'hour']].values.astype(float)

    ecpc = np.sum(train_data[:, 0]) / np.sum(train_data[:, 2])
    origin_ctr = np.sum(train_data[:, 0]) / len(train_data)
    avg_mprice = np.sum(train_data[:, 2]) / len(train_data)

    return train_data, test_data, ecpc, origin_ctr, avg_mprice


def reward_func(reward_type, fab_clks, hb_clks, fab_cost, hb_cost):
    if fab_clks >= hb_clks and fab_cost < hb_cost:
        r = 5
    elif fab_clks >= hb_clks and fab_cost >= hb_cost:
        r = 1
    elif fab_clks < hb_clks and fab_cost >= hb_cost:
        r = -5
    else:
        r = -2.5

    if reward_type == 'op':
        return r / 1000
    elif reward_type == 'nop':
        return r
    elif reward_type == 'nop_2.0':
        return fab_clks / 1000
    else:
        return fab_clks


'''
1458
437520 447493
30309883.0 30297100.0
395.0 356.0

3358
237844 335310
23340047.0 32515709.0
197.0 307.0

3386
412275 392901
32967478.0 31379459.0
344.0 355.0

3427
379524 390398
30918866.0 31654042.0
282.0 313.0

'''

if __name__ == '__main__':
    campaign_id = '1458/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)

    train_data, test_data, ecpc, origin_ctr, avg_mprice = get_dataset(args)

    setup_seed(args.seed)

    log_dirs = [args.save_log_dir, args.save_log_dir + args.campaign_id]
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    param_dirs = [args.save_param_dir, args.save_param_dir + args.campaign_id]
    for param_dir in param_dirs:
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + str(args.campaign_id).strip('/') + args.model_name + '_output.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    submission_path = args.data_path + args.dataset_name + args.campaign_id + args.model_name + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    device = torch.device(args.device)  # 指定运行设备

    logger.info(campaign_id)
    logger.info('RL model ' + args.model_name + ' has been training')
    logger.info(args)

    actions = np.array(list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10)))

    rl_model = get_model(args, device)
    B = args.budget * args.budget_para[0]

    hb_clk_dict = {}
    for para in actions:
        bid_datas = generate_bid_price(train_data[:, 1] * para / origin_ctr)
        res_ = bid_main(bid_datas, train_data, B)
        hb_clk_dict.setdefault(para, res_[0])

    hb_base = sorted(hb_clk_dict.items(), key=lambda x: x[1])[-1][0]

    train_losses = []

    logger.info('para:{}, budget:{}, base bid: {}'.format(args.budget_para[0], B, hb_base))
    logger.info('\tclks\treal_clks\tbids\timps\tcost')

    start_time = datetime.datetime.now()

    clk_index, ctr_index, mprice_index, hour_index = 0, 1, 2, 3

    ep_train_records = []
    ep_test_records = []
    ep_test_actions = []
    for ep in range(args.episodes):
        if ep % 10 == 0:
            test_records = [0, 0, 0, 0, 0]
            tmp_test_state = [1, 0, 0, 0]
            init_test_state = [1, 0, 0, 0]
            test_actions = [0 for _ in range(24)]
            test_rewards = 0
            budget = B
            hour_t = 0
            for t in range(24):
                if budget > 0:
                    hour_datas = test_data[test_data[:, hour_index] == t]

                    state = torch.tensor(init_test_state).float() if not t else torch.tensor(tmp_test_state).float()

                    action = rl_model.choose_action(state.unsqueeze(0))[0, 0].item()
                    test_actions[t] = action
                    bid_datas = generate_bid_price((hour_datas[:, ctr_index] * hb_base / origin_ctr) / (1 + action))
                    res_ = bid_main(bid_datas, hour_datas, budget)

                    # win_clks, real_clks, bids, imps, cost

                    test_records = [test_records[i] + res_[i] for i in range(len(test_records))]
                    budget -= res_[-1]

                    hb_bid_datas = generate_bid_price(hour_datas[:, ctr_index] * hb_base / origin_ctr)
                    res_hb = bid_main(hb_bid_datas, hour_datas, budget)

                    r_t = reward_func(args.reward_type, res_[0], res_hb[0], res_[3], res_hb[3])
                    test_rewards += r_t

                    left_hour_ratio = (23 - t) / 23 if t <= 23 else 0
                    # avg_budget_ratio, cost_ratio, ctr, win_rate
                    next_state = [(budget / B) / left_hour_ratio if left_hour_ratio else (budget / B),
                                  res_[4] / B,
                                  res_[0] / res_[3] if res_[3] else 0,
                                  res_[3] / res_[2] if res_[2] else 0]
                    tmp_test_state = next_state

                    hour_t += 1
            ep_test_records.append([ep] + test_records + [test_rewards])
            ep_test_actions.append(test_actions)
            print(ep, 'test', test_records, test_rewards)

        budget = B
        tmp_state = [1, 0, 0, 0]
        init_state = [1, 0, 0, 0]
        train_records = [0, 0, 0, 0, 0]
        # win_clks, real_clks, bids, imps, cost
        # win_clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
        critic_loss = 0

        done = 0
        for t in range(24):
            if budget > 0:
                hour_datas = train_data[train_data[:, hour_index] == t]

                state = torch.tensor(init_state).float() if not t else torch.tensor(tmp_state).float()

                action = rl_model.choose_action(state.unsqueeze(0))[0, 0].item()

                bid_datas = generate_bid_price((hour_datas[:, ctr_index] * (hb_base / origin_ctr)) / (1 + action))
                res_ = bid_main(bid_datas, hour_datas, budget)
                # win_clks, real_clks, bids, imps, cost

                train_records = [train_records[i] + res_[i] for i in range(len(train_records))]
                budget -= res_[-1]

                left_hour_ratio = (23 - t) / 23 if t <= 23 else 0

                if (not left_hour_ratio) or (budget <= 0):
                     done = 1

                # avg_budget_ratio, cost_ratio, ctr, win_rate
                next_state = [(budget / B) / left_hour_ratio if left_hour_ratio else (budget / B),
                              res_[4] / B,
                              res_[0] / res_[3] if res_[3] else 0,
                              res_[3] / res_[2] if res_[2] else 0]
                tmp_state = next_state

                hb_bid_datas = generate_bid_price(hour_datas[:, ctr_index] * hb_base / origin_ctr)
                res_hb = bid_main(hb_bid_datas, hour_datas, budget)

                r_t = reward_func(args.reward_type, res_[0], res_hb[0], res_[3], res_hb[3])

                transitions = torch.cat([state, torch.tensor([action]).float(),
                                         torch.tensor(next_state).float(),
                                         torch.tensor([done]).float(), torch.tensor([r_t]).float()], dim=-1).unsqueeze(
                    0).to(device)

                rl_model.store_transition(transitions)

                if rl_model.memory.memory_counter >= args.rl_batch_size:
                    critic_loss = rl_model.learn()
        if ep % 10 == 0:
            ep_train_records.append([ep] + train_records + [critic_loss])

        # print('train', records, critic_loss)

    train_record_df = pd.DataFrame(data=ep_train_records,
                                   columns=['ep', 'clks', 'real_clks', 'bids', 'imps', 'cost', 'loss'])
    train_record_df.to_csv(submission_path + 'fab_train_records_' + args.reward_type + str(args.budget_para[0]) + '.csv', index=None)

    test_record_df = pd.DataFrame(data=ep_test_records,
                                  columns=['ep', 'clks', 'real_clks', 'bids', 'imps', 'cost', 'loss'])
    test_record_df.to_csv(submission_path + 'fab_test_records_' + args.reward_type + str(args.budget_para[0]) + '.csv', index=None)

    test_action_df = pd.DataFrame(data=ep_test_actions)
    test_action_df.to_csv(submission_path + 'fab_test_actions_' + args.reward_type + str(args.budget_para[0]) + '.csv')