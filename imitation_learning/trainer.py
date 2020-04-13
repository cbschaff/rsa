"""Defines trainer and network for MNIST."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import time
from dl import logger, Checkpointer, nest
from dl.util import StatefulSampler
from dl.modules import DiagGaussian
from torch.utils.data import DataLoader
import os


@gin.configurable
class BCNet(nn.Module):
    """Imitation network."""

    def __init__(self, ob_shape, action_shape, nunits):
        """Init."""
        super().__init__()
        self.nunits = nunits
        self.fc1 = nn.Linear(ob_shape, self.nunits)
        self.fc2 = nn.Linear(self.nunits, self.nunits)
        self.dist = DiagGaussian(self.nunits, action_shape)
        self.ob_shape = ob_shape

    def forward(self, ob):
        """Forward."""
        if ob.shape[-1] > self.ob_shape:
            # HACK. While iterating on the project, I changed the
            # observation space of the environments. To use human
            # data collected before that, I manually crop out
            # newer observation data.
            ob = ob[:, :self.ob_shape]
        net = F.relu(self.fc1(ob))
        net = F.relu(self.fc2(net))
        return self.dist(net)


@gin.configurable(blacklist=['datafile'])
class DemonstrationData(object):
    def __init__(self, datafile, mean=None, std=None):
        data = torch.load(datafile)
        self.obs = data['obs']
        self.actions = data['actions']
        self.n = self.obs.shape[0]
        if mean is None or std is None:
            self.mean = self.obs.mean(dim=0)
            self.std = self.obs.std(dim=0)
        else:
            self.mean = torch.from_numpy(mean)
            self.std = torch.from_numpy(std)
        self.obs = (self.obs - self.mean) / (self.std + 1e-5)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


@gin.configurable(blacklist=['logdir'])
class BCTrainer(object):
    """Behavioral cloning."""

    def __init__(self, logdir, model, opt, datafile, batch_size, num_workers,
                 gpu=True):
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        self.data = DemonstrationData(datafile)
        self.sampler = StatefulSampler(self.data, shuffle=True)
        self.dtrain = DataLoader(self.data, sampler=self.sampler,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        self._diter = None
        self.t = 0
        self.epochs = 0
        self.batch_size = batch_size

        self.device = torch.device('cuda:0' if gpu and torch.cuda.is_available()
                                   else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.opt = opt(self.model.parameters())

    def step(self):
        # Get batch.
        if self._diter is None:
            self._diter = self.dtrain.__iter__()
        try:
            batch = self._diter.__next__()
        except StopIteration:
            self.epochs += 1
            self._diter = None
            return self.epochs
        batch = nest.map_structure(lambda x: x.to(self.device), batch)

        # compute loss
        ob, ac = batch
        self.model.train()
        loss = -self.model(ob).log_prob(ac).mean()

        logger.add_scalar('train/loss', loss.detach().cpu().numpy(),
                          self.t, time.time())

        # update model
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # increment step
        self.t += min(len(self.data) - (self.t % len(self.data)),
                      self.batch_size)
        return self.epochs

    def evaluate(self):
        """Evaluate model."""
        self.model.eval()

        nll = 0.
        with torch.no_grad():
            for batch in self.dtrain:
                ob, ac = nest.map_structure(lambda x: x.to(self.device), batch)
                nll += -self.model(ob).log_prob(ac).sum()
            avg_nll = nll / len(self.data)

            logger.add_scalar('train/NLL', nll, self.epochs, time.time())
            logger.add_scalar('train/AVG_NLL', avg_nll, self.epochs,
                              time.time())

    def save(self):
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['opt'] = self.opt.state_dict()
        state_dict['sampler'] = self.sampler.state_dict(self._diter)
        state_dict['t'] = self.t
        state_dict['epochs'] = self.epochs
        self.ckptr.save(state_dict, self.t)

    def load(self, t=None):
        state_dict = self.ckptr.load()
        if state_dict is None:
            self.t = 0
            self.epochs = 0
            return self.epochs
        self.model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['opt'])
        self.sampler.load_state_dict(state_dict['sampler'])
        self.t = state_dict['t']
        self.epochs = state_dict['epochs']
        if self._diter is not None:
            self._diter.__del__()
            self._diter = None

    def close(self):
        """Close data iterator."""
        if self._diter is not None:
            self._diter.__del__()
            self._diter = None
