import os
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import time
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss
from collections import namedtuple

import copy
from lib import plotting, utils
import itertools

batch_size = 32
state_size = 3
frame_len = 1
replay_buffer_size = 20000
Target_update = 1000
replay_start_size = 500

ctx = mx.cpu() # Enables gpu if available, if not, set it to mx.cpu()
seed = 2023

def build_sharedNet(env, s_net):
    with s_net.name_scope():
        s_net.add(gluon.nn.Dense(32, activation='sigmoid'))
    return s_net

def build_individualNet(env, i_net):
    with i_net.name_scope():
        i_net.add(gluon.nn.Dense(32, activation='sigmoid'))
        i_net.add(gluon.nn.Dense(env.action_space.n))
    return i_net

def m2dqn(env,num_episodes,exploration,gamma,lr, epsilon,epsilon_decay,decay_steps,initial_run,sNet,iNet):

    num_action = env.action_space.n 

    mx.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)                             

    metalr = 0.5

    ######### Train the model #########

    if initial_run == True:

        sNet = build_sharedNet(env, gluon.nn.Sequential())
        iNet = build_individualNet(env, gluon.nn.Sequential())

        sNet.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
        iNet.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)       

        X = nd.random.uniform(shape=(32,3))
        Y = iNet(sNet(X))
    
    sNet_trainer = gluon.Trainer(sNet.collect_params(), 'adam', {'learning_rate': lr})
    iNet_trainer = gluon.Trainer(iNet.collect_params(), 'adam', {'learning_rate': lr})

    targetsNet = copy.deepcopy(sNet) 
    targetiNet = copy.deepcopy(iNet)

    originaliNet = copy.deepcopy(iNet)

    print('--'*18)
    print('replay_buffer_size(rbz):' + str(replay_buffer_size))
    print('replay_start_size(rsz):' + str(replay_start_size))
    print('batch_size(bs):' + str(batch_size))
    print('Target_update(tu):' + str(Target_update))   
    print('--'*18)

    l2loss = gluon.loss.L2Loss(batch_axis=0)
    replay_memory = Replay_Buffer(replay_buffer_size) 

    batch_state = nd.empty((batch_size,frame_len,state_size), ctx)
    batch_state_next = nd.empty((batch_size,frame_len,state_size), ctx)
    batch_reward = nd.empty((batch_size),ctx)
    batch_action = nd.empty((batch_size),ctx)
    batch_done = nd.empty((batch_size),ctx)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    total_steps = 0
    epsilon_i = epsilon

    for  epis_count in range(num_episodes):
        state = env.reset()
        state = state_trans(state)
        start_time = time.time()
        eps = 0

        for t in itertools.count(): #while True:
        
            # annealing_count += 1

            if exploration == 'e_greedy':
                data = nd.array(state.reshape([1,frame_len,state_size]),ctx)
                action, eps = e_greedy(num_action, sNet, iNet, data, epsilon_i)
            elif exploration == 'boltzmann':
                action,  act_prob= boltzmann(num_action,  sNet, iNet, state)

            next_state, reward, done, _ = env.step(action)
            next_state = state_trans(next_state)
            replay_memory.push(state,action,next_state,reward,done)
            total_steps += 1
            state = next_state
            
            # Update statistics
            stats.episode_rewards[epis_count] += reward
            stats.episode_lengths[epis_count] = t            

            # Train
            if total_steps > replay_start_size:        
                batch = replay_memory.sample(batch_size)
                loss = 0

                for j in range(batch_size):
                    batch_state[j] = batch[j].state.astype('float32')
                    batch_state_next[j] = batch[j].next_state.astype('float32')
                    batch_reward[j] = batch[j].reward
                    batch_action[j] = batch[j].action
                    batch_done[j] = batch[j].done

                with autograd.record():
                    # get the Q(s,a)
                    all_current_q_value = iNet(sNet(batch_state))
                    main_q_value = nd.pick(all_current_q_value,batch_action,1)                        
                    # get next action from main network
                    all_next_q_value = targetiNet(targetsNet(batch_state_next)).detach() # only get gradient of main network
                    max_action = nd.argmax(all_current_q_value, axis=1)                        
                    # then get its Q value from target network
                    target_q_value = nd.pick(all_next_q_value, max_action).detach()
                    target_q_value = batch_reward + (1-batch_done)*gamma *target_q_value
                    # record loss
                    loss = loss + nd.mean(l2loss(main_q_value, target_q_value))
                        
                loss.backward()
                sNet_trainer.step(batch_size)
                iNet_trainer.step(batch_size)

            # Save the model and update Target model
            if total_steps > replay_start_size:
                if total_steps % Target_update == 0 :
                    targetsNet = copy.deepcopy(sNet)
                    targetiNet = copy.deepcopy(iNet)
                    print('target network parameters replaced')
        
            if done == True:
                break
        
        # apply a simple decaying epsilon during the first decay_steps
        if epis_count < decay_steps:
            epsilon_i = epsilon_i * epsilon_decay

        end_time = time.time()
        duration = 1000. * (end_time - start_time)
        
        if exploration == 'e_greedy':
            results = 'eps% d, reward %d, egreedy %f' % (epis_count, stats.episode_rewards[epis_count],epsilon_i)
        elif exploration == 'boltzmann':
            results = 'eps% d, reward %d, boltzmann' % (epis_count, stats.episode_rewards[epis_count])
        if epis_count % 1 == 0:
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(results)
    
    originaliNet[0].weight.data()[:]  = originaliNet[0].weight.data()[:] + (originaliNet[0].weight.data()[:] - iNet[0].weight.data()[:])* metalr
    originaliNet[1].weight.data()[:]  = originaliNet[1].weight.data()[:] + (originaliNet[1].weight.data()[:] - iNet[1].weight.data()[:])* metalr
    iNet[0].weight.data()[:] = originaliNet[0].weight.data()[:]
    iNet[1].weight.data()[:] = originaliNet[1].weight.data()[:]
    
    return sNet,iNet, stats


### Replay buffer
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done'))
class Replay_Buffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.replay_buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.replay_buffer_size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

def state_trans(state):
    state = state.replace('[','').replace(']','').split(',')
    state = np.array((state[0],state[1],state[2]))
    return state

def e_greedy(num_action,  sNet, iNet, state, epsilon):
    sample = random.random()
    # epsilon greedy policy
    if sample < epsilon:
        action = np.random.choice(range(num_action))
    else:
        action = int(nd.argmax(iNet(sNet(state)),axis=1).asscalar())
    return action, epsilon

def boltzmann(num_action,  sNet, iNet, state):
    state_q_array = (iNet(sNet(nd.array(state.reshape([1,frame_len,state_size]),ctx)))).asnumpy()
    # Subtract the maximum value to prevent overflow
    state_q = state_q_array - np.max(state_q_array)
    # Calculate the selection probability for each action
    act_prob  = np.exp(state_q)/np.sum(np.exp(state_q))
    # Select the next action based on the calculated probability
    action = np.random.choice(range(num_action),  p = act_prob[0])
    return action, act_prob[0]