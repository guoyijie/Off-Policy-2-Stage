import numpy as np
import torch
from model import Nominator, UserRep, ItemRep
from loss import loss_ips
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnsembleCritic(nn.Module):

    def __init__(self, hidden1=256, hidden2=128):
        super(EnsembleCritic, self).__init__()
        self.user_rep = UserRep()
        self.item_rep = ItemRep()
        input_dim = self.user_rep.rep_dim + self.item_rep.rep_dim
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, 1), nn.Sigmoid())
        self.l2 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, 1), nn.Sigmoid())
        self.l3 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, 1), nn.Sigmoid())
        self.l4 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, 1), nn.Sigmoid()) 

    def forward(self, user_feats, item_feats):
        users = self.user_rep(**user_feats)
        items = self.item_rep(**item_feats)
        num_users = users.shape[0]
        num_items = items.shape[0]
        users = users.reshape(num_users,1,-1).repeat(1,num_items,1).view(num_users*num_items, -1)
        items = items.reshape(num_items,-1).repeat(num_users,1)
        inputs = torch.cat([users, items], dim=1)
        all_qs = torch.cat([self.l1(inputs).unsqueeze(0), self.l2(inputs).unsqueeze(0), self.l3(inputs).unsqueeze(0), self.l4(inputs).unsqueeze(0),], 0)
        all_qs = all_qs.view(4,num_users, num_items)
        return all_qs
"""

class EnsembleCritic(nn.Module):
    def __init__(self):
        super(EnsembleCritic, self).__init__()
        self.item_rep = ItemRep()
        self.user_rep = UserRep()
        self.l1 = nn.Linear(self.user_rep.rep_dim, self.item_rep.rep_dim)
        self.l2 = nn.Linear(self.user_rep.rep_dim, self.item_rep.rep_dim)
        self.l3 = nn.Linear(self.user_rep.rep_dim, self.item_rep.rep_dim)
        self.l4 = nn.Linear(self.user_rep.rep_dim, self.item_rep.rep_dim)

    def forward(self, user_feats, item_feats):
        users = self.user_rep(**user_feats)
        users1 = self.l1(F.relu(users)).unsqueeze(2) 
        users2 = self.l2(F.relu(users)).unsqueeze(2)
        users3 = self.l3(F.relu(users)).unsqueeze(2)
        users4 = self.l4(F.relu(users)).unsqueeze(2)
        items = self.item_rep(**item_feats)
        items = torch.unsqueeze(items, 0).expand(users.size(0), -1,
                                                     -1)  # (c, h) -> (b, c, h)
        logit1 = torch.bmm(items, users1).squeeze()
        logit2 = torch.bmm(items, users2).squeeze()
        logit3 = torch.bmm(items, users3).squeeze()
        logit4 = torch.bmm(items, users4).squeeze()
        return torch.cat([logit1.unsqueeze(0), logit2.unsqueeze(0), logit3.unsqueeze(0), logit4.unsqueeze(0),], 0)
"""


class Continuation_Q(object):
    def __init__(self, lr, behavior=None):
        #self.mode = mode
        self.actor = Nominator().to(device)
        self.actor.set_binary(False)
        self.actor_opt = torch.optim.Adagrad(self.actor.parameters(), lr=0.01, weight_decay=1e-4)

        self.critic = EnsembleCritic().to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=0.01)

        self.behavior = Nominator().to(device)
        self.behavior.set_binary(False)
        self.behavior_opt = torch.optim.Adagrad(self.behavior.parameters(), lr=0.01, weight_decay=1e-4)

    def train(self, user_feats, val_item_feats, item_ids, item_probs, item_rewards, c=0, epoch=0):

        # train behavior
        logits = self.behavior(user_feats, val_item_feats)
        probs = torch.gather(F.softmax(logits, dim=1),1, item_ids.view(-1,1)).view(-1)
        behavior_loss = -probs.mean()
        self.behavior_opt.zero_grad()
        behavior_loss.backward()
        self.behavior_opt.step()
        #train critic values
       
        all_q_values = self.critic(user_feats, val_item_feats)
        all_q_values = torch.gather(all_q_values, 2, item_ids.view(1,-1,1).repeat(4,1,1))
        critic_loss = nn.BCELoss()(all_q_values[0].view(-1), item_rewards)+nn.BCELoss()(all_q_values[1].view(-1), item_rewards)+nn.BCELoss()(all_q_values[2].view(-1), item_rewards)+nn.BCELoss()(all_q_values[3].view(-1), item_rewards)
        print('critic loss', critic_loss)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        # train actor
        logits = self.actor(user_feats, val_item_feats)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        all_q_values = self.critic(user_feats, val_item_feats)
        actor_loss = (-log_probs*all_q_values.mean(0).detach()*probs.detach()).sum(1).mean()       
        #print('actor loss', actor_loss)
        betas = F.softmax(self.behavior(user_feats, val_item_feats), dim=1)
        kl_divergence = torch.sum(probs*torch.log(probs/(betas+1e-6)+1e-6), 1)
        if epoch>10:
            actor_loss += c*kl_divergence.mean()
        else:
            actor_loss = kl_divergence.mean()
        #upper_limit = 10
        #lower_limit = 0.01
        #importance_weight = torch.gather(F.softmax(logits.detach(), dim=1),1, item_ids.view(-1,1)).view(-1) / item_probs
        #importance_weight = torch.clamp(importance_weight, lower_limit, upper_limit)
        #importance_weight /= torch.mean(importance_weight)
        #actor_loss = -torch.mean(torch.gather(F.log_softmax(logits, dim=1),1, item_ids.view(-1,1)).view(-1) * importance_weight * all_q_values.mean(0).detach())
        #probs = F.softmax(logits, dim=1)
        #log_probs = F.log_softmax(logits, dim=1)
        #if self.mode=='likelihood':
        #    actor_loss = (-log_probs*q_values*probs.detach()).mean()
            #print('actor loss', actor_loss)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()



class Continuation(object):
    def __init__(self, lr, behavior):
        
        self.actor = Nominator().to(device)
        self.actor.set_binary(False)
        self.actor_opt = torch.optim.Adagrad(self.actor.parameters(), lr=lr, weight_decay=1e-4)

        self.critic = EnsembleCritic().to(device)

        self.behavior = Nominator().to(device)
        self.behavior.set_binary(False)
        self.behavior_opt = torch.optim.Adagrad(self.behavior.parameters(), lr=0.01, weight_decay=1e-4)
        #self.actor.load_state_dict(self.behavior.state_dict())

    def train(self, user_feats, val_item_feats, item_ids, item_probs, item_rewards, c, epoch):
        # train behavior
        logits = self.behavior(user_feats, val_item_feats)
        probs = torch.gather(F.softmax(logits, dim=1),1, item_ids.view(-1,1)).view(-1)
        behavior_loss = -probs.mean()
        print(behavior_loss)
        self.behavior_opt.zero_grad()
        behavior_loss.backward()
        self.behavior_opt.step()       
 
        #train actor
        logits = self.actor(user_feats, val_item_feats)
        upper_limit = 10
        lower_limit = 0.01
        importance_weight = torch.gather(F.softmax(logits.detach(), dim=1),1, item_ids.view(-1,1)).view(-1) / item_probs
        importance_weight = torch.clamp(importance_weight, lower_limit, upper_limit)
        importance_weight /= torch.mean(importance_weight)
        actor_loss = -torch.mean(torch.gather(F.log_softmax(logits, dim=1),1, item_ids.view(-1,1)).view(-1) * importance_weight * item_rewards)
        probs = F.softmax(logits, dim=1)
        betas = F.softmax(self.behavior(user_feats, val_item_feats), dim=1)
        kl_divergence = torch.sum(probs*torch.log(probs/(betas+1e-6)+1e-6), 1)
        if epoch>10:
            actor_loss += c*kl_divergence.mean()
        else:
            actor_loss = kl_divergence.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


class IPS(object):
    def __init__(self, lr):
        self.actor = Nominator().to(device)
        self.actor.set_binary(False)
        self.actor_opt = torch.optim.Adagrad(self.actor.parameters(), lr=lr, weight_decay=1e-4)

        self.critic = EnsembleCritic().to(device)

    def train(self, user_feats, val_item_feats, item_ids, item_probs, item_rewards):
        logits = self.actor(user_feats, val_item_feats)
        upper_limit = 10
        lower_limit = 0.01
        importance_weight = torch.gather(F.softmax(logits.detach(), dim=1),1, item_ids.view(-1,1)).view(-1) / item_probs
        importance_weight = torch.clamp(importance_weight, lower_limit, upper_limit)
        importance_weight /= torch.mean(importance_weight)
        actor_loss = -torch.mean(torch.gather(F.log_softmax(logits, dim=1),1, item_ids.view(-1,1)).view(-1) * importance_weight * item_rewards)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
