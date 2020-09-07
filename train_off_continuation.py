import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import MovieLensDataset, SyntheticMovieLensDataset
import Continuation
from metric import Evaluator
from model import ImpressionSimulator, Nominator, Ranker
from six.moves import cPickle as pickle
from torch.utils.data import DataLoader, Subset
from torchnet.meter import AUCMeter

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=int, default=1, help="Verbose.")
parser.add_argument("--seed", type=int, default=1, help="Random seed.")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
parser.add_argument("--kl_weight", type=float, default=10.0, help="KL weight.")
parser.add_argument("--decay", type=float, default=0.9, help="KL weight decay.")
parser.add_argument("--log", type=str, default='results/tmp', help="log dir.")
args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

filepath = "ml-1m/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MovieLensDataset(filepath, device=device)

NUM_ITEMS = dataset.NUM_ITEMS
NUM_YEARS = dataset.NUM_YEARS
NUM_GENRES = dataset.NUM_GENRES

NUM_USERS = dataset.NUM_USERS
NUM_OCCUPS = dataset.NUM_OCCUPS
NUM_AGES = dataset.NUM_AGES
NUM_ZIPS = dataset.NUM_ZIPS

simulator_path = os.path.join(filepath, "simulator.pt")
simulator = ImpressionSimulator(use_impression_feats=True)
simulator.load_state_dict(torch.load(simulator_path))
simulator.to(device)
simulator.eval()

# create a torch dataset class that adopt the simulator and generate the synthetic dataset
synthetic_data_path = os.path.join(filepath, "full_impression_feats.pt")
syn = SyntheticMovieLensDataset(
    filepath, simulator_path, synthetic_data_path, device=device)

logging_policy_path = os.path.join(filepath, "logging_policy.pt")
logging_policy = Nominator()
logging_policy.load_state_dict(torch.load(logging_policy_path))
logging_policy.to(device)

def generate_bandit_samples(logging_policy, syn, k=5):
    """Generates partial-labeled bandit samples with the logging policy.

    Arguments:
        k: The number of items to be sampled for each user.
    """
    logging_policy.set_binary(False)
    with torch.no_grad():
        feats = {}
        feats["user_feats"] = syn.user_feats
        feats["item_feats"] = syn.item_feats
        feats = syn.to_device(feats)
        probs = F.softmax(logging_policy(**feats), dim=1)

    sampled_users = []
    sampled_actions = []
    sampled_probs = []
    sampled_rewards = []
    for i in range(probs.size(0)):
        sampled_users.append([i] * k)
        sampled_actions.append(
            torch.multinomial(probs[i], k).cpu().numpy().tolist())
        sampled_probs.append(
            probs[i, sampled_actions[-1]].cpu().numpy().tolist())
        sampled_rewards.append(syn.impression_feats["labels"][[
            i * probs.size(1) + j for j in sampled_actions[-1]
        ]].numpy().tolist())
    return np.array(sampled_users).reshape(-1), np.array(
        sampled_actions).reshape(-1), np.array(sampled_probs).reshape(
            -1), np.array(sampled_rewards).reshape(-1)


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

u, a, p, r = generate_bandit_samples(
    logging_policy, syn,
    k=5)  # u: user, a: item, p: logging policy probability, r: reward/label

ev = Evaluator(u[r > 0], a[r > 0], simulator, syn)

all_user_feats = syn.to_device(syn.user_feats)
all_item_feats = syn.to_device(syn.item_feats)
all_impression_feats = syn.to_device({
    "real_feats":
    torch.mean(
        syn.impression_feats["real_feats"].view(NUM_USERS, NUM_ITEMS),
        dim=1).view(-1, 1)
})

# Split validation/test users.
num_val_users = 2000
val_user_list = list(range(0, num_val_users))
test_user_list = list(range(num_val_users, NUM_USERS))

test_item_feats = all_item_feats
test_user_feats = syn.to_device(
    {key: value[test_user_list]
     for key, value in all_user_feats.items()})
test_impression_feats = syn.to_device({
    key: value[test_user_list]
    for key, value in all_impression_feats.items()
})

val_item_feats = all_item_feats
val_user_feats = syn.to_device(
    {key: value[val_user_list]
     for key, value in all_user_feats.items()})
val_impression_feats = syn.to_device(
    {key: value[val_user_list]
     for key, value in all_impression_feats.items()})

ranker_path = os.path.join(filepath, "ranker.pt")
ranker = Ranker()
ranker.load_state_dict(torch.load(ranker_path))
ranker.to(device)
ranker.eval()
ranker.set_binary(False)

#u = u[r > 0]
#a = a[r > 0]
#p = p[r > 0]
#r = r[r > 0]

batch_size = 256
check_metric = "Precision@10"

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
policy = Continuation.Continuation(lr=args.lr, behavior=logging_policy)

# Init results.
best_epoch = 0
best_result = 0.0
val_results, test_results = [], []

c=args.kl_weight
for epoch in range(100):
    print("---epoch {}---".format(epoch))
    for step in range(len(u) // batch_size):
        idx = np.random.randint(0, len(u), size=batch_size)
        item_ids = torch.LongTensor(a[idx]).to(device)
        item_probs = torch.FloatTensor(p[idx]).to(device)
        item_rewards = torch.FloatTensor(r[idx]).to(device)

        user_ids = u[idx]
        user_feats = {
            key: value[user_ids]
            for key, value in syn.user_feats.items()
        }
        user_feats = syn.to_device(user_feats)
        policy.train(user_feats, val_item_feats, item_ids, item_probs, item_rewards, c, epoch)
    
    with torch.no_grad():
        policy.actor.eval()
        policy.critic.eval()

        logits = policy.actor(all_user_feats, all_item_feats)
        print("1 stage", ev.one_stage_eval(logits))
        print("2 stage", ev.two_stage_eval(logits, ranker))

        # Evaluate ranking metrics on validation users.
        logits = policy.actor(val_user_feats, val_item_feats)
        one_stage_results = ev.one_stage_ranking_eval(logits, val_user_list)
        print("1 stage (val)", one_stage_results)
        two_stage_results = ev.two_stage_ranking_eval(logits, ranker,
                                                      val_user_list)
        print("2 stage (val)", two_stage_results)
        val_results.append((one_stage_results, two_stage_results))
        # Log best epoch
        if two_stage_results[check_metric] > best_result:
            best_epoch = epoch
            best_result = two_stage_results[check_metric]
        # Evaluate ranking metrics on test users.
        logits = policy.actor(test_user_feats, test_item_feats)
        one_stage_results = ev.one_stage_ranking_eval(logits, test_user_list)
        print("1 stage (test)", one_stage_results)
        two_stage_results = ev.two_stage_ranking_eval(logits, ranker,
                                                      test_user_list)
        print("2 stage (test)", two_stage_results)
        test_results.append((one_stage_results, two_stage_results))

        policy.actor.train()
        policy.critic.train()
    if epoch > 10:
        c = c*args.decay    

print("Best validation epoch: {}".format(best_epoch))
print("Best validation stage results\n 1 stage: {}\n 2 stage: {}".format(
    val_results[best_epoch][0], val_results[best_epoch][1]))
print("Best test results\n 1 stage: {}\n 2 stage: {}".format(
    test_results[best_epoch][0], test_results[best_epoch][1]))

np.save(args.log+'/metric.npy', (best_epoch, val_results, test_results))
