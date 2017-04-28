# This code is aiming to inspect the ability of Evolution Strategies.
# See the following.
# Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" https://arxiv.org/abs/1703.03864
# https://blog.openai.com/evolution-strategies/

import numpy as np
import gym
from gym import wrappers

gym_target = 'CartPole-v0'
env = gym.make(gym_target)

n_actions = env.action_space.n
n_perturbations = 4
idim = env.observation_space.shape[0]
hdim = 3
odim = n_actions
sigma = 0.5
alpha = 0.05
n_iter = 300

class Model:
    def __init__(self, idim, hdim, odim):
        def rand_array(shape):
            return np.random.random(shape) - 0.5
        self.w0 = rand_array((hdim, idim))
        self.b0 = np.zeros(hdim)
        self.w1 = rand_array((odim ,hdim))
        self.b1 = np.zeros(odim)
    def reset_perturbations(self, n_perturbations):
        def gen_pert(param, n):
            shape = (n,) + param.shape
            return np.random.normal(0, 1.0, shape)
        n = n_perturbations
        self.w0p = gen_pert(self.w0, n)
        self.b0p = gen_pert(self.b0, n)
        self.w1p = gen_pert(self.w1, n)
        self.b1p = gen_pert(self.b1, n)
        self.n_perturbations = n_perturbations
    def evaluate(self, f, sigma):
        w0t = self.w0
        b0t = self.b0
        w1t = self.w1
        b1t = self.b1
        self.f = np.zeros(self.n_perturbations, dtype=np.float32)
        self.sigma = sigma
        for i in range(self.n_perturbations):
            self.w0 = w0t + sigma * self.w0p[i]
            self.b0 = b0t + sigma * self.b0p[i]
            self.w1 = w1t + sigma * self.w1p[i]
            self.b1 = b1t + sigma * self.b1p[i]
            self.f[i] = f(self)
        self.w0 = w0t
        self.b0 = b0t
        self.w1 = w1t
        self.b1 = b1t
    def calculate_grads(self, clip=None):
        w0g = np.zeros(self.w0.shape, dtype=np.float32)
        b0g = np.zeros(self.b0.shape, dtype=np.float32)
        w1g = np.zeros(self.w1.shape, dtype=np.float32)
        b1g = np.zeros(self.b1.shape, dtype=np.float32)
        for i in range(self.n_perturbations):
            w0g += self.f[i] * self.w0p[i]
            b0g += self.f[i] * self.b0p[i]
            w1g += self.f[i] * self.w1p[i]
            b1g += self.f[i] * self.b1p[i]
        c = 1.0 / (self.n_perturbations * self.sigma)
        self.w0g = w0g * c
        self.b0g = b0g * c
        self.w1g = w1g * c
        self.b1g = b1g * c
        if not(clip is None):
            self.w0g = np.clip(self.w0g, -clip, clip)
            self.b0g = np.clip(self.b0g, -clip, clip)
            self.w1g = np.clip(self.w1g, -clip, clip)
            self.b1g = np.clip(self.b1g, -clip, clip)
    def update(self, alpha):
        self.w0 += self.w0g * alpha
        self.b0 += self.b0g * alpha
        self.w1 += self.w1g * alpha
        self.b1 += self.b1g * alpha
    def __call__(self, x):
        h = np.dot(self.w0, x) + self.b0
        z = np.dot(self.w1, h) + self.b1
        return z

def learn(model):
    f = evaluate
    for i in range(n_iter):
        model.reset_perturbations(n_perturbations)
        model.evaluate(f, sigma)
        model.calculate_grads(clip=5.0)
        model.update(alpha)

def evaluate(model):
    def softmax(x):
        y = np.exp(x)
        return y / np.sum(y)

    def choice_action(model, obs):
        probs = softmax(model(obs))
        return np.argmax(probs)

    acc = 0
    obs = env.reset()
    gamma = 1.05
    mul = 1
    while True:
        if monitor:
            env.render()
        action = choice_action(model, obs)
        obs, reward, done, info = env.step(action)
        acc += mul
        mul *= gamma
        if done:
            return acc

monitor = False

model = Model(idim, hdim, odim)
learn(model)

if monitor:
    env = wrappers.Monitor(env, './monitor', force=True)

print(evaluate(model))
