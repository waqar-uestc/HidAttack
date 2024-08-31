import numpy as np
import torch
import random
from time import time
from hyper import parameters
from datasets import load_dataset
from client import HidClient
from server import HidServer
from Hidattack import Malicious, AttackCollection
import  pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,
    log_loss,
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def epsilon_greedy_selection(attack_rewards, epsilon=0.1):
    """Epsilon-greedy strategy for selecting the next attack option."""
    if np.random.random() < epsilon:  # Exploration
        return np.random.choice(len(attack_rewards))
    else:  # Exploitation
        return np.argmax(attack_rewards)


def HidAttack(att):
    parameters.attack = att
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in parameters.__dict__.items()])
    print("Arguments: %s " % args_str)

    emb_interval = 5
    t0 = time()
    m_item, all_train_ind, all_test_ind, items_popularity = load_dataset(parameters.path + parameters.dataset)
    _, target_items = torch.Tensor(-items_popularity).topk(1)
    target_items = target_items.tolist()  # Select the least popular item as the target item
    server = HidServer(m_item, parameters.dim, eval(parameters.layers)).to(parameters.device)
    clients = []
    for train_ind, test_ind in zip(all_train_ind, all_test_ind):
        clients.append(
            HidClient(train_ind, test_ind, target_items, m_item, parameters.dim).to(parameters.device)
        )

    malicious_clients_limit = int(len(clients) * parameters.clients_limit)
    if parameters.attack == 'A3' or parameters.attack == 'A4':
        for _ in range(malicious_clients_limit):
            clients.append(Malicious(target_items, m_item, parameters.dim).to(parameters.device))
    elif parameters.attack == 'A2':
        for _ in range(malicious_clients_limit):
            clients.append(AttackCollection(target_items, m_item, parameters.dim).to(parameters.device))
    elif parameters.attack == 'A1':
        for _ in range(malicious_clients_limit):
            train_ind = [i for i in target_items]
            for __ in range(parameters.items_limit - len(target_items)):
                item = np.random.randint(m_item)
                while item in train_ind:
                    item = np.random.randint(m_item)
                train_ind.append(item)
            clients.append(AttackCollection(train_ind, m_item, parameters.dim).to(parameters.device))

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))

    # The evaluation of performance and HidAttack
    train_log = []
    t1 = time()
    test_result, target_result = server.eval_(clients)
    
    print("Iteration 0, Recorded BEC_Loss = 0, Recommendation Indicators (HR@20, Prec@20, NDCG@20, MRR@20) are (%.7f, %.7f, %.7f, %.7f) " % tuple(test_result))
    print("\t Targeted Item Exposure Ratio (ER@5, ER@10, ER@20, ER@30) at current iteration are (%.7f, %.7f, %.7f, %.7f) " % tuple(target_result) + " [%.1fs]" % (time() - t1))
    
    try:
        for epoch in range(1, parameters.epochs + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []
            for i in range(0, len(rand_clients), parameters.batch_size):
                batch_clients_idx = rand_clients[i: i + parameters.batch_size]
                loss = server.train_(clients, batch_clients_idx, epoch, emb_interval)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()
            t2 = time()
            test_result, target_result = server.eval_(clients)
            print("Attack Iterations %d, Recorded BEC_Loss = %.5f [%.1fs]" % (epoch, total_loss, t2 - t1))
            print("\t -----------Recommendation Indicators (HR@20, Prec@20, NDCG@20, MRR@20) are (%.7f, %.7f, %.7f, %.7f)" % tuple(test_result))
            print("\t -----------Targeted Item Exposure Ratio (ER@5, ER@10, ER@20, ER@30) at current iteration are (%.7f, %.7f, %.7f, %.7f)" % tuple(target_result) + " [%.1fs]" % (time() - t2))
            
            train_log.append([epoch,total_loss, *test_result,*target_result ])            
            df = pd.DataFrame(train_log, columns=['Iteration', 'BEC_Loss', 'HR', 'Precision', 'NDCG', 'MRR', 'ER@5', 'ER@10', 'ER@20', 'ER@30'])
            df.to_csv("Results_" + parameters.dataset +"_"+ parameters.attack +"_"+ str(parameters.clients_limit) + ".csv", index=False)
        average_ER = df[['ER@5', 'ER@10', 'ER@20', 'ER@30']].mean().mean()
        return average_ER

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    setup_seed(20220110)
  
    attack_cluster = ['A4', 'A3', 'A2', 'A1']
    num_attacks = len(attack_cluster)
    
    attack_rewards = [0] * num_attacks  # Stores cumulative rewards for each attack
    attack_counts = [0] * num_attacks   # Tracks the number of times each attack was selected
    
    num_iterations = 300
    epsilon = 0.1  # Exploration factor for the epsilon-greedy strategy
    
    for i in range(num_iterations):
        selected_attack_idx = epsilon_greedy_selection(attack_rewards, epsilon)
        selected_attack = attack_cluster[selected_attack_idx]
        
        reward = HidAttack(selected_attack)
        attack_counts[selected_attack_idx] += 1
        # Update the average reward for the selected attack
        attack_rewards[selected_attack_idx] += (reward - attack_rewards[selected_attack_idx]) / attack_counts[selected_attack_idx]
    
    best_attack_idx = np.argmax(attack_rewards)
    best_attack = attack_cluster[best_attack_idx]
    
    print("The final output need to be process as per Read me instructions")