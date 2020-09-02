# adapted from https://github.com/Kaixhin/Rainbow

import torch
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)
simple_blank_trans = Transition(0, torch.zeros(8, dtype=torch.float32), None, 0, False)


class SegmentTree:
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.sum_tree = np.zeros((2 * size - 1,), dtype=np.float32)
        self.data = np.array([None] * size)
        self.max = 1

    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    def update(self, index, value):
        self.sum_tree[index] = value
        self._propagate(index, value)
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data
        self.update(self.index + self.size - 1, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def total(self):
        return self.sum_tree[0]

    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return self.sum_tree[index], data_index, index

    def get(self, data_index):
        return self.data[data_index % self.size]


class ReplayMemory:
    def __init__(self, capacity, history_length, discount, multi_step, priority_weight, priority_exponent):
        self.device = torch.device("cuda:0")
        self.capacity = capacity
        self.history = history_length
        self.discount = discount
        self.n = multi_step
        self.priority_weight = priority_weight
        self.priority_exponent = priority_exponent
        self.t = 0
        self.transitions = SegmentTree(capacity)

    def _append(self, state, action, reward, terminal):
        self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)
        self.t = 0 if terminal else self.t + 1

    def append(self, state, action, reward, terminal):
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device("cpu"))
        self._append(state, action, reward, terminal)

    def _return_transition(self, idx, blank_transition):
        transition = np.array([None] * (self.history + self.n))
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):
            if transition[t + 1].timestep == 0:
                transition[t] = blank_transition
            else:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
            else:
                transition[t] = blank_transition
        return transition

    def _get_transition(self, idx):
        return self._return_transition(idx, blank_trans)

    def _return_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)
            prob, idx, tree_idx = self.transitions.find(sample)
            if (self.transitions.index - idx) % self.capacity > self.n \
                    and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True

        transition = self._get_transition(idx)
        state = torch.stack([trans.state for trans in transition[:self.history]]) \
            .to(self.device).to(dtype=torch.float32)
        next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]) \
            .to(self.device).to(dtype=torch.float32)
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
        r = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))],
                         dtype=torch.float32, device=self.device)
        nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal],
                                   dtype=torch.float32, device=self.device)

        return prob, idx, tree_idx, state, action, r, next_state, nonterminal

    def _get_sample_from_segment(self, segment, i):
        prob, idx, tree_idx, state, action, r, next_state, nonterminal = self._return_sample_from_segment(segment, i)
        state = state.div_(255)
        next_state = next_state.div_(255)
        return prob, idx, tree_idx, state, action, r, next_state, nonterminal

    def sample(self, batch_size):
        p_total = self.transitions.total()
        segment = p_total / batch_size
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
        probs = np.array(probs, dtype=np.float32) / p_total
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]


class SimpleReplayMemory(ReplayMemory):
    def __init__(self, capacity, history_length, discount, multi_step, priority_weight, priority_exponent):
        super(SimpleReplayMemory, self).__init__(capacity, history_length, discount, multi_step,
                                                 priority_weight, priority_exponent)

    def append(self, state, action, reward, terminal):
        state = state[-1].to(dtype=torch.float32, device=torch.device("cpu"))
        self._append(state, action, reward, terminal)

    def _get_transition(self, idx):
        return self._return_transition(idx, simple_blank_trans)

    def _get_sample_from_segment(self, segment, i):
        return self._return_sample_from_segment(segment, i)
